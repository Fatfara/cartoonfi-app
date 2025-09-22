import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import io

def preserve_facial_features(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Enhanced cartoon effect that preserves facial features and details
    """
    
    # Convert to OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Step 1: Moderate smoothing (preserve more details)
    smooth = img_cv.copy()
    iterations = max(3, int(5 * intensity))
    
    for i in range(iterations):
        # Use smaller kernel sizes to preserve features
        d = 9 + i
        sigma_color = 50 + i * 10
        sigma_space = 50 + i * 10
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)
    
    # Step 2: Enhanced edge detection that preserves facial features
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Create multiple edge layers
    # Fine details (eyes, nose, mouth)
    edges_fine = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=7,
        C=4
    )
    
    # Medium details (face outline, major features)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_medium = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=6
    )
    
    # Combine edges intelligently
    edges = cv2.bitwise_or(edges_fine, edges_medium)
    
    # Clean up edges while preserving important features
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small)
    
    # Step 3: Intelligent color quantization with more colors
    data = smooth.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    k = max(8, int(12 - intensity * 2))  # More colors to preserve features
    
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS
    )
    
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(smooth.shape)
    
    # Step 4: Smart edge blending that preserves features
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Create a softer mask
    edges_float = edges.astype(np.float32) / 255.0
    edges_mask = np.stack([edges_float] * 3, axis=2)
    
    # Blend more conservatively to keep features
    cartoon = quantized.astype(np.float32)
    cartoon = cartoon * (0.85 + 0.15 * edges_mask)
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 5: Selective smoothing (avoid over-smoothing faces)
    # Apply less smoothing to preserve facial details
    cartoon = cv2.bilateralFilter(cartoon, 5, 30, 30)
    
    return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

def detailed_cartoon_effect(img: Image.Image, style: str = "balanced") -> Image.Image:
    """
    Detailed cartoon effect with better feature preservation
    """
    
    # Pre-enhance for better feature detection
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if style == "balanced":
        # Balanced approach - good detail retention
        smooth = img_cv.copy()
        for _ in range(4):
            smooth = cv2.bilateralFilter(smooth, 11, 60, 60)
        
        # Multi-scale edge detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Fine edges for facial features
        edges1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=7, C=3
        )
        
        # Canny for strong edges
        edges2 = cv2.Canny(gray, 30, 100)
        
        # Combine
        edges = cv2.bitwise_or(edges1, edges2)
        
        k_colors = 10
        
    elif style == "detailed":
        # More detailed preservation
        smooth = img_cv.copy()
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, 9, 50, 50)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        edges1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=5, C=2
        )
        
        edges2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=7, C=3
        )
        
        edges3 = cv2.Canny(gray, 20, 80)
        
        # Combine all edges
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        
        k_colors = 12
        
    elif style == "smooth":
        # Smoother but still detailed
        smooth = img_cv.copy()
        for _ in range(6):
            smooth = cv2.bilateralFilter(smooth, 13, 80, 80)
        
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=5
        )
        
        k_colors = 8
    
    # Clean edges appropriately
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Slight edge softening
    edges = cv2.GaussianBlur(edges, (1, 1), 0)
    
    # Color quantization
    data = smooth.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(smooth.shape)
    
    # Gentle edge application
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_mask = edges.astype(np.float32) / 255.0
    edges_mask = np.stack([edges_mask] * 3, axis=2)
    
    cartoon = quantized.astype(np.float32)
    cartoon = cartoon * (0.8 + 0.2 * edges_mask)
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Light final smoothing
    cartoon = cv2.bilateralFilter(cartoon, 7, 40, 40)
    
    cartoon_pil = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    
    # Appropriate color enhancement
    if style == "detailed":
        enhancer = ImageEnhance.Color(cartoon_pil)
        cartoon_pil = enhancer.enhance(1.2)
    else:
        enhancer = ImageEnhance.Color(cartoon_pil)
        cartoon_pil = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(cartoon_pil)
        cartoon_pil = enhancer.enhance(1.1)
    
    return cartoon_pil

def advanced_portrait_cartoon(img: Image.Image, face_preservation: float = 0.8) -> Image.Image:
    """
    Advanced cartoon specifically designed for portraits with facial feature preservation
    """
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    original = img_cv.copy()
    
    # Step 1: Detect facial regions (simple skin tone detection)
    # Convert to HSV for better skin tone detection
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Define skin tone range (approximate)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Step 2: Apply different processing to face and non-face areas
    
    # For facial areas - gentle processing
    smooth_face = img_cv.copy()
    for _ in range(int(3 * face_preservation)):
        smooth_face = cv2.bilateralFilter(smooth_face, 9, 50, 50)
    
    # For non-facial areas - stronger processing
    smooth_bg = img_cv.copy()
    for _ in range(6):
        smooth_bg = cv2.bilateralFilter(smooth_bg, 15, 100, 100)
    
    # Combine using skin mask
    skin_mask_3d = np.stack([skin_mask] * 3, axis=2) / 255.0
    smooth = (smooth_face * skin_mask_3d + smooth_bg * (1 - skin_mask_3d)).astype(np.uint8)
    
    # Step 3: Edge detection with facial area consideration
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Fine edges for facial features
    edges_fine = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=5, C=2
    )
    
    # General edges
    edges_general = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=4
    )
    
    # Use fine edges in facial areas, general edges elsewhere
    edges = np.where(skin_mask[..., np.newaxis] > 0, edges_fine[..., np.newaxis], edges_general[..., np.newaxis]).squeeze()
    
    # Clean edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Adaptive color quantization
    data = smooth.reshape((-1, 3))
    data = np.float32(data)
    
    # More colors for better facial detail
    k = int(10 + face_preservation * 4)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS
    )
    
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(smooth.shape)
    
    # Step 5: Intelligent edge blending
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_mask = edges.astype(np.float32) / 255.0
    edges_mask = np.stack([edges_mask] * 3, axis=2)
    
    cartoon = quantized.astype(np.float32)
    cartoon = cartoon * (0.75 + 0.25 * edges_mask)
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 6: Selective final smoothing
    # Less smoothing on facial areas
    final_smooth = cv2.bilateralFilter(cartoon, 5, 30, 30)
    cartoon = (final_smooth * (1 - skin_mask_3d * 0.7) + cartoon * (skin_mask_3d * 0.7)).astype(np.uint8)
    
    cartoon_pil = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    
    # Enhanced color processing
    enhancer = ImageEnhance.Color(cartoon_pil)
    cartoon_pil = enhancer.enhance(1.25)
    
    enhancer = ImageEnhance.Contrast(cartoon_pil)
    cartoon_pil = enhancer.enhance(1.1)
    
    return cartoon_pil

def create_streamlit_app():
    """Enhanced Streamlit app with better cartoon generation"""
    
    st.set_page_config(
        page_title="🎨 Enhanced Cartoon Generator", 
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Enhanced Cartoon Generator")
    st.write("Generate detailed cartoon images that preserve important facial features!")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Generation Settings")
        
        method = st.selectbox(
            "Cartoon Method",
            options=["Portrait Cartoon (Best for Faces)", "Detailed Cartoon", "Feature Preserving", "Custom"],
            index=0
        )
        
        if method == "Portrait Cartoon (Best for Faces)":
            face_preservation = st.slider("Face Detail Level", 0.5, 1.0, 0.8, 0.1)
            
        elif method == "Detailed Cartoon":
            style = st.selectbox(
                "Detail Level", 
                options=["balanced", "detailed", "smooth"], 
                index=0
            )
            
        elif method == "Feature Preserving":
            intensity = st.slider("Cartoon Intensity", 0.5, 2.0, 1.0, 0.1)
            
        elif method == "Custom":
            smoothing = st.slider("Smoothing Strength", 1, 8, 4)
            colors = st.slider("Number of Colors", 6, 16, 10)
            edge_strength = st.slider("Edge Strength", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("### 🎯 Method Guide")
        st.markdown("**Portrait Cartoon**: Best for face photos")
        st.markdown("**Detailed Cartoon**: Preserves more details")
        st.markdown("**Feature Preserving**: Balanced approach")
        st.markdown("**Custom**: Full manual control")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)
            st.info(f"Size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.header("🎨 Enhanced Cartoon Result")
        
        if uploaded_file is not None:
            with st.spinner("🎨 Creating enhanced cartoon..."):
                try:
                    # Generate cartoon based on selected method
                    if method == "Portrait Cartoon (Best for Faces)":
                        cartoon_img = advanced_portrait_cartoon(image, face_preservation)
                        
                    elif method == "Detailed Cartoon":
                        cartoon_img = detailed_cartoon_effect(image, style)
                        
                    elif method == "Feature Preserving":
                        cartoon_img = preserve_facial_features(image, intensity)
                        
                    elif method == "Custom":
                        # Custom implementation
                        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Apply custom smoothing
                        smooth = img_cv.copy()
                        for _ in range(smoothing):
                            smooth = cv2.bilateralFilter(smooth, 11, 60, 60)
                        
                        # Custom edge detection
                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                        edges = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                            blockSize=7, C=edge_strength
                        )
                        
                        # Custom quantization
                        data = smooth.reshape((-1, 3))
                        data = np.float32(data)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
                        _, labels, centers = cv2.kmeans(
                            data, colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
                        )
                        centers = np.uint8(centers)
                        quantized = centers[labels.flatten()].reshape(smooth.shape)
                        
                        # Combine
                        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                        edges_mask = edges.astype(np.float32) / 255.0
                        edges_mask = np.stack([edges_mask] * 3, axis=2)
                        
                        cartoon = quantized.astype(np.float32) * (0.8 + 0.2 * edges_mask)
                        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
                        
                        cartoon_img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
                    
                    # Display result
                    st.image(cartoon_img, caption="Enhanced Cartoon Result", use_container_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    cartoon_img.save(buf, format="PNG", quality=100)
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="💾 Download Enhanced Cartoon",
                        data=byte_im,
                        file_name="enhanced_cartoon.png",
                        mime="image/png"
                    )
                    
                    st.success("✨ Enhanced cartoon generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👆 Upload an image to generate an enhanced cartoon!")
    
    # Tips section
    st.markdown("---")
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For Portrait Photos:**
        - Use **"Portrait Cartoon"** method for best facial feature preservation
        - Set Face Detail Level to 0.8-1.0 for maximum detail retention
        - Well-lit photos with clear facial features work best
        
        **For General Images:**
        - Use **"Detailed Cartoon"** with "balanced" or "detailed" settings
        - **"Feature Preserving"** works well for images with important details
        
        **Parameter Guidelines:**
        - **Face Detail Level**: Higher = more facial detail preserved
        - **Cartoon Intensity**: Lower = more realistic, Higher = more stylized
        - **Edge Strength**: Higher = more defined edges and details
        """)

if __name__ == "__main__":
    create_streamlit_app()