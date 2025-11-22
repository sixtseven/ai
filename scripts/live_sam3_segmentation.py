import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import time
import argparse

# --- Configuration ---
SAM3_MODEL_ID = "facebook/sam3"
# You might need to change '0' to '1' or '2' if you have multiple webcams
WEBCAM_INDEX = 0 
# --- Configuration ---


def main():
    parser = argparse.ArgumentParser(description='SAM 3 Live Webcam Segmentation Demo')
    parser.add_argument('--prompt', type=str, required=True, 
                        help='The text concept prompt for SAM 3 (e.g., "cup", "keyboard", "person").')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on (e.g., cuda, cpu).')
    args = parser.parse_args()
    
    concept_prompt = args.prompt
    device = torch.device(args.device)

    print(f"Loading SAM 3 model '{SAM3_MODEL_ID}' on {device}...")
    try:
        # Load the model and processor (weights are auto-downloaded if authenticated)
        processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
        model = Sam3Model.from_pretrained(SAM3_MODEL_ID).to(device)
    except Exception as e:
        print(f"Failed to load SAM 3. Ensure you have access to '{SAM3_MODEL_ID}' on Hugging Face and are logged in.")
        print(f"Error: {e}")
        return

    print("--- Model Loaded ---")
    print(f"Concept Prompt: '{concept_prompt}'.")
    print("Press 'Q' or 'ESC' to exit the demo.")
    
    # 1. Initialize Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
        return

    # Helper function to generate a unique color for each instance mask
    def generate_random_color(i):
        # Generate a semi-random, consistent color based on the instance index
        r = ((i * 37) % 256)
        g = ((i * 101) % 256)
        b = ((i * 73) % 256)
        return (r, g, b)

    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Prepare Frame for SAM 3
        # SAM 3 expects PIL Image input
        image_pil = Image.fromarray(frame_rgb)
        
        # Prepare inputs with the text prompt for Promptable Concept Segmentation (PCS)
        inputs = processor(images=image_pil, text=concept_prompt, return_tensors="pt").to(device)
        
        # 3. Run SAM 3 Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 4. Post-process the Results
        # Get the mask instances from the outputs
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,           # Confidence threshold for detection
            mask_threshold=0.5,      # Binary mask threshold
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results['masks'] # [N, H, W] tensor of boolean masks
        
        # 5. Visualize Results
        overlay = frame_bgr.copy()
        alpha = 0.4
        
        for i, mask in enumerate(masks):
            mask_np = mask.cpu().numpy()
            color = generate_random_color(i)
            
            # Create a colored overlay layer
            colored_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
            colored_mask[mask_np] = color
            
            # Blend the mask with the original frame
            cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display FPS and Prompt
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, f"Prompt: {concept_prompt}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Objects: {len(masks)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show the result
        cv2.imshow('SAM 3 Live Concept Segmentation', overlay)

        # Exit on 'q' or 'ESC' press
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()