import argparse

from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer

from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=None,
                        help="Image paths or URLs")
    parser.add_argument("--video", type=str, default=None,
                        help="Video path or URL")
    parser.add_argument("--prompt", type=str, default="Describe this image.",
                        help="User prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--inject-embedding", action="store_true", help="Inject custom embeddings")
    parser.add_argument("--embedding-path", type=str, default="", help="Embedding path")
    parser.add_argument("--model-name", type=str, default="mobilint/Qwen3-VL-8B-Instruct", help="Model name")
    parser.add_argument("--text-mxq-path", type=str, default="./example.mxq", help="Text MXQ path")
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        text_mxq_path=args.text_mxq_path,
        trust_remote_code=True,
    )
    if args.inject_embedding:
        MobilintModelMixin._inject_custom_embeddings(model, args.embedding_path)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)

    content = []
    images = None
    videos = None

    if args.video:
        content.append({"type": "video", "video": args.video})
        videos = [args.video]
    elif args.images:
        content.extend([{"type": "image", "image": img} for img in args.images])
        images = args.images

    content.append({"type": "text", "text": args.prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")

    # --- First turn ---
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, streamer=streamer)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    first_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n")

    # --- Second turn ---
    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user", "content": "What else can you tell me about it?"})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")

    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, streamer=streamer)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    second_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n=== First turn ===")
    print(first_response)
    print("\n=== Second turn ===")
    print(second_response)


if __name__ == "__main__":
    main()
