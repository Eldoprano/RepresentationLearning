from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from repe.pattern_reader import PatternReader
from repe.pattern_control import PatternRepControl

def main():
    # 1. Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("your_model_name")
    tokenizer = AutoTokenizer.from_pretrained("your_model_name")
    
    # 2. Initialize pattern reader
    pattern_reader = PatternReader(
        model=model,
        tokenizer=tokenizer,
        rep_token=-1,  # Use last token for LAT
        direction_method="pca"
    )
    
    # 3. Training data to learn the pattern
    train_data = [
        "I will tell the truth: The sky is blue.",
        "I will be honest: Water is wet.",
        "Let me be truthful: Birds fly in the sky."
    ]
    train_labels = [1] * len(train_data)  # 1 for honest statements

    # 4. Learn the pattern using LAT
    directions = pattern_reader.get_directions(
        train_data,
        train_labels=train_labels
    )

    # 5. Test pattern detection
    test_data = [
        "I will lie: The sky is green.",
        "Let me be honest: The Earth is round."
    ]
    detection_results = pattern_reader(test_data)

    print("Pattern Detection Results:")
    for text, result in zip(test_data, detection_results):
        honesty_score = result[list(result.keys())[0]]  # Get first layer's score
        print(f"Text: {text}")
        print(f"Honesty Score: {honesty_score}\n")

    # 6. Control text generation using the pattern
    controller = PatternRepControl(
        model=model,
        tokenizer=tokenizer,
        control_section="full",  # Control entire generation
        control_scope="full"     # Apply to all sentences
    )

    # Prepare activations for control
    layers = list(directions.keys())
    activations = {}
    control_strength = 1.0
    
    for layer in layers:
        activations[layer] = (
            control_strength * 
            directions[layer] * 
            pattern_reader.direction_signs[layer]
        ).to(model.device)

    # Generate text with and without control
    prompt = "Tell me about the weather today:"

    print("Generating without control:")
    baseline = controller.generate(
        prompt,
        activations={},  # Empty dict for no control
        max_new_tokens=50,
        do_sample=False
    )
    print(baseline)

    print("\nGenerating with honesty control:")
    controlled = controller.generate(
        prompt, 
        activations=activations,
        activation_scale=1.0,
        max_new_tokens=50,
        do_sample=False
    )
    print(controlled)

if __name__ == "__main__":
    main()
