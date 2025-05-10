from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import torch

def batch_encode_sentences(
    sentences: list[str], 
    model_name: str = "meta-llama/Meta-Llama-3.1-70B",
    batch_size: int = 8 # You can adjust batch size based on your VRAM
    ) -> torch.Tensor | None:
    """
    Encodes a list of sentences into vector embeddings using a specified Hugging Face model.

    Args:
        sentences (list[str]): A list of sentences to encode.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "meta-llama/Meta-Llama-3.1-70B".
        batch_size (int): The number of sentences to process in a single batch.

    Returns:
        torch.Tensor | None: A tensor containing the sentence embeddings (on CPU).
                             Each row corresponds to a sentence embedding.
                             Returns None if an error occurs.
    """
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load tokenizer and model
        # For large models like Llama 70B, you might need to specify torch_dtype for memory efficiency
        # e.g., torch_dtype=torch.float16 or torch.bfloat16 if your hardware supports it.
        # You might also need device_map="auto" for multi-GPU or CPU offloading.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32, # Use float16 on GPU if available
            # device_map="auto" # Uncomment if you need model parallelism / CPU offloading
        ).to(device)
        model.eval() # Set model to evaluation mode

        # Add padding token if it doesn't exist (common for Llama models)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
                print(f"Tokenizer `pad_token` was None, set to `eos_token`: {tokenizer.eos_token}")
            else:
                # Add a new pad token if eos_token is also missing (less common)
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                print("Added a new `pad_token` [PAD] to the tokenizer.")

        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # Tokenize sentences
            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 512
            ).to(device)

            # Get model outputs (no_grad to disable gradient calculations for inference)
            with torch.no_grad():
                outputs = model(**inputs)

            # Mean Pooling:
            # Multiply token embeddings by attention mask to zero out padding tokens
            # Then sum and divide by the number of non-padding tokens.
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            
            batch_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(batch_embeddings.cpu()) # Move to CPU before storing

        return torch.cat(all_embeddings, dim=0)

    except Exception as e:
        print(f"An error occurred during batch encoding: \n{e}")
        if "meta-llama" in model_name and ("You must have access to it and be authenticated to access it. Please log in." in str(e) or "gated" in str(e)):
            print("Please ensure you are logged in to Hugging Face CLI (`huggingface-cli login`) "
                  "and have accepted the model's terms on its Hugging Face Hub page.")
        return None

if __name__ == '__main__':
    login(token="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")
    example_sentences = [
        "This is the first sentence for testing.",
        "Here is another, slightly longer example sentence for encoding purposes.",
        "Large language models are transforming AI."
    ]

    print(f"Encoding {len(example_sentences)} sentences...")

    # For demonstration, using a smaller, publicly accessible model.
    # Replace with "meta-llama/Meta-Llama-3.1-70B" if you have access and resources.
    embeddings = batch_encode_sentences(example_sentences, model_name="meta-llama/Llama-3.1-8B")
    # embeddings = batch_encode_sentences(example_sentences, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=2)
    # embeddings = batch_encode_sentences(example_sentences, batch_size=2)

    if embeddings is not None:
        print(f"\nSuccessfully generated embeddings. Shape: {embeddings.shape}")
        print("First embedding (first 5 values):", embeddings[0][:5])
    else:
        print("\nFailed to generate embeddings.")
