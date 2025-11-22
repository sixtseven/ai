import torch
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def overlay_masks(image: Image.Image, masks: torch.Tensor, alpha: float = 0.5):
    """
    Overlay segmentation masks on the input image using different colors.
    image: PIL.Image
    masks: Tensor (N, H, W) with binary masks
    returns: PIL.Image (RGBA)
    """

    if masks is None or len(masks) == 0:
        print("No masks detected. Returning original image.")
        return image

    image = image.convert("RGBA")

    masks_np = (masks.cpu().numpy() * 255).astype("uint8")
    num_masks = masks_np.shape[0]

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(num_masks, 1))
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_masks)]

    for mask_np, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask_np)
        overlay = Image.new("RGBA", image.size, color + (0,))

        alpha_mask = mask_img.point(lambda p: int(p * alpha))
        overlay.putalpha(alpha_mask)

        image = Image.alpha_composite(image, overlay)

    return image


def run_sam3(image_path: str, output_path: str):
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Running inference...")
    state = processor.set_image(image)

    # text prompt
    text_prompt = "suitcase, backpack, luggage, bag"

    # get SAM3 output
    output = processor.set_text_prompt(
        state=state,
        prompt=text_prompt
    )

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    print(f"Detected {len(masks)} objects")
    if len(masks) > 0:
        print("Scores:", scores)

    overlay_image = overlay_masks(image, masks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_image)
    axes[1].set_title("Luggage / Bags (SAM3)")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 luggage detector with overlay.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save output visualization")

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    run_sam3(args.image_path, args.output)


if __name__ == "__main__":
    main()
