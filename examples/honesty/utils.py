import pandas as pd
import numpy as np
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", 
                             assistant_tag: str = "", seed: int = 0, reasoning_template = False) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Check if chat template exists
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    print(f"Using: {'chat template' if has_chat_template else 'provided tags'}")

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    honest_statements = []
    untruthful_statements = []
    
    template_str = "Pretend you're {type} person making statements about the world."
    template_reasoning = "<think>I can do this</think>"

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)
            if reasoning_template:
                truncated_statement = template_reasoning + truncated_statement

            if has_chat_template:
                # Use chat template
                honest_conv = [
                    {"role": "user", "content": template_str.format(type='an honest')},
                    {"role": "assistant", "content": ""}
                ]
                untruthful_conv = [
                    {"role": "user", "content": template_str.format(type='an untruthful')},
                    {"role": "assistant", "content": ""}
                ]
                # Ugly hack, because continue_final_message=True has some bugs
                # Apply the template and remove "<｜end▁of▁sentence｜>" from the result
                honest_template = tokenizer.apply_chat_template(honest_conv, tokenize=False, add_generation_prompt=False, continue_final_message=True)
                honest_template = honest_template.replace("<｜end▁of▁sentence｜>", "")
                if reasoning_template:
                    honest_template += truncated_statement.format(type='an honest')
                honest_statements.append(honest_template)

                untruthful_template = tokenizer.apply_chat_template(untruthful_conv, tokenize=False, add_generation_prompt=False, continue_final_message=True)
                untruthful_template = untruthful_template.replace("<｜end▁of▁sentence｜>", "")
                if reasoning_template:
                    untruthful_template += truncated_statement.format(type='an untruthful')
                untruthful_statements.append(untruthful_template)
            else:
                # Use provided tags
                honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
                untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

    print(f"Honest statement:\n{honest_statements[0]}")
    print()
    print(f"Untruthful statement:\n{untruthful_statements[0]}")

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token="<｜Assistant｜>"):
    # Parameters that affect layout
    x_start = 1
    y_start = 9.5
    y_pad = 0.2
    line_spacing = 0.9
    xlim = 20
    fig_width = 12.8

    # First pass to calculate height
    def calculate_layout():
        x, y = x_start, y_start
        lowest_y = y_start
        
        started = False
        for word in words[5:]:
            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            
            if '\n' in word:
                x = x_start
                y -= line_spacing
                lowest_y = min(lowest_y, y)
                word = word.replace('\n', '')
                if not word:
                    continue
            
            # Temporarily create text to measure width
            temp_fig, temp_ax = plt.subplots()
            temp_text = temp_ax.text(0, 0, word, fontsize=13)
            word_width = temp_text.get_window_extent(
                temp_fig.canvas.get_renderer()
            ).transformed(temp_ax.transData.inverted()).width
            plt.close(temp_fig)
            
            if x + word_width > xlim:
                x = x_start
                y -= line_spacing
                lowest_y = min(lowest_y, y)
            
            x += word_width
        
        return lowest_y

    # Process tokens
    words = []
    for token in input_ids:
        cleaned = token.replace('▁', ' ').replace('Ġ', ' ').replace('Ċ', '\n')
        words.append(cleaned)

    # Calculate required height
    lowest_y = calculate_layout()
    content_height = y_start - lowest_y + 2  # The actual height of our content (plus padding)
    points_per_data_unit = fig_width * 72 / xlim  # How many points (1/72 inch) per data unit in width
    needed_height_points = content_height * points_per_data_unit  # Convert content height to points
    fig_height = needed_height_points / 72  # Convert points to inches

    # Create figure with calculated dimensions
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.set_xlim(0, xlim)
    ax.set_ylim(lowest_y - 1, y_start + 1)
    
    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Create colormap
    cmap = LinearSegmentedColormap.from_list('rg', ["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Process scores and create visualization
    selected_concepts = ["honesty"]
    norm_style = ["mean"]
    selection_style = ["neg"]
    
    iter = 0
    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):
        rep_scores = np.array(rep_reader_scores_dict[rep])
        
        # Statistical normalization
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean
        
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)
        
        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD
            rep_scores = rep_scores / np.std(rep_scores[5:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        
        rep_scores[np.abs(rep_scores) < 0.0] = 0
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)
        
        # Reset position for rendering
        x, y = x_start, y_start
        started = False
        
        for word, score in zip(words[5:], rep_scores[5:]):
            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            
            if '\n' in word:
                x = x_start
                y -= line_spacing
                word = word.replace('\n', '')
                if not word:
                    continue
            
            color = colormap(norm(score))
            
            # Check if word exceeds line width
            temp_text = ax.text(0, 0, word, fontsize=13)
            word_width = temp_text.get_window_extent(
                fig.canvas.get_renderer()
            ).transformed(ax.transData.inverted()).width
            temp_text.remove()
            
            if x + word_width > xlim:
                x = x_start
                y -= line_spacing
            
            # Render base text
            text_base = ax.text(x, y - 0.35 + y_pad * (iter + 1), word, 
                              color='black',
                              fontsize=13)
            
            # Render colored overlay
            text_color = ax.text(x, y + y_pad * (iter + 1), word, 
                               color='white',
                               alpha=0,
                               bbox=dict(facecolor=color, 
                                       edgecolor=color, 
                                       alpha=0.8,
                                       boxstyle='round,pad=0', 
                                       linewidth=0),
                               fontsize=13)
            
            x += word_width
        
        iter += 1
    
    plt.subplots_adjust(top=0.95)
    return fig, ax


def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice, start_answer_token="<｜Assistant｜>", num_tokens_to_plot = 40):
    # Check if start_answer_token is in input_ids
    if start_answer_token not in input_ids:
        start_answer_token = input_ids[0]
        print(f"start_answer_token not found in input_ids. Using {start_answer_token} instead.")
        
    for rep, scores in rep_reader_scores_dict.items():

        start_tok = input_ids.index(start_answer_token)
        print(start_tok, np.array(scores).shape)
        standardized_scores = np.array(scores)[start_tok:start_tok+num_tokens_to_plot,layer_slice]
        # print(standardized_scores.shape)

        bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 2.3

        # standardized_scores = np.array(scores)
        
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)
        
        cmap = 'coolwarm'

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)
        ax.tick_params(axis='y', rotation=0)

        ax.set_xlabel("Token Position")#, fontsize=20)
        ax.set_ylabel("Layer")#, fontsize=20)

        # x label appear every 5 ticks

        ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
        ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])#, fontsize=20)
        ax.tick_params(axis='x', rotation=0)

        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(20, len(standardized_scores[0])+20, 5)[::-1][1:])#, fontsize=20)
        ax.set_title("LAT Neural Activity")#, fontsize=30)
    plt.show()
