import streamlit as st
from pathlib import Path

def setup_page_config():
    st.set_page_config(
        page_title="AI Style Transfer Studio",
        page_icon="🎨",
        layout="wide"
    )

def apply_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background-color: #1f2937;
        }
        .stMarkdown {
            color: #f3f4f6;
        }
        .stButton > button {
            background-color: #6366F1;
            color: white;
        }
        .stButton > button:hover {
            background-color: #4F46E5;
        }
        .dark-theme {
            background-color: #111827;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #374151;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="dark-theme" style="text-align: center;">
        <h1>🎨 AI Style Transfer Studio</h1>
        <h3>Transform your ideas into artistic masterpieces</h3>
    </div>
    """, unsafe_allow_html=True)

def render_controls(style_names):
    with st.sidebar:
        st.markdown("## 🎯 Controls")
        
        prompt = st.text_area(
            "What would you like to create?",
            placeholder="e.g., a soccer player celebrating a goal",
            height=100
        )
        
        selected_style = st.radio(
            "Choose Your Style",
            style_names,
            index=0
        )
        
        return prompt, selected_style

def render_image_columns(base_image=None, enhanced_image=None):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Style")
        if base_image:
            st.image(base_image, use_column_width=True)
    
    with col2:
        st.markdown("### Color Enhanced")
        if enhanced_image:
            st.image(enhanced_image, use_column_width=True)

def render_example_gallery():
    st.markdown("""
    <div class="dark-theme">
        <h2>🎆 Example Gallery</h2>
        <p>Compare original and enhanced versions for each style:</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        output_dir = Path("Outputs")
        original_dir = output_dir
        enhanced_dir = output_dir / "Color_Enhanced"

        if enhanced_dir.exists():
            original_images = {
                Path(f).stem.split('_example')[0]: f 
                for f in original_dir.glob("*.webp") 
                if '_example' in f.name
            }
            enhanced_images = {
                Path(f).stem.split('_example')[0]: f 
                for f in enhanced_dir.glob("*.webp") 
                if '_example' in f.name
            }

            styles = [
                ("ronaldo", "Ronaldo Style"),
                ("canna_lily", "Canna Lily"),
                ("three_stooges", "Three Stooges"),
                ("pop_art", "Pop Art"),
                ("bird_style", "Bird Style")
            ]

            for style_key, style_name in styles:
                if style_key in original_images and style_key in enhanced_images:
                    st.markdown(f"### {style_name}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(
                            str(original_images[style_key]),
                            caption="Original",
                            use_column_width=True
                        )
                    with col2:
                        st.image(
                            str(enhanced_images[style_key]),
                            caption="Color Enhanced",
                            use_column_width=True
                        )
                    st.markdown("<hr>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading example gallery: {str(e)}")

def render_info_sections():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="dark-theme">
            <h2>🎨 Style Guide</h2>
            <table>
                <tr>
                    <th>Style</th>
                    <th>Best For</th>
                </tr>
                <tr>
                    <td><strong>Dhoni Style</strong></td>
                    <td>Cricket scenes, sports action, victory celebrations</td>
                </tr>
                <tr>
                    <td><strong>Mickey Mouse Style</strong></td>
                    <td>Cartoon characters, playful scenes, whimsical art</td>
                </tr>
                <tr>
                    <td><strong>Balloon Style</strong></td>
                    <td>Festive scenes, colorful celebrations, light and airy compositions</td>
                </tr>
                <tr>
                    <td><strong>Lion King Style</strong></td>
                    <td>Animal portraits, majestic scenes, dramatic landscapes</td>
                </tr>
                <tr>
                    <td><strong>Rose Flower Style</strong></td>
                    <td>Floral art, romantic scenes, delicate compositions</td>
                </tr>
            </table>
            <em>Choose the style that best matches your creative vision</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dark-theme">
            <h2>🔍 Color Enhancement Technology</h2>
            <p>Our advanced color processing uses distance loss to maximize the distinction between color channels, 
            resulting in more vibrant and visually striking images. This technique helps to:</p>
            <ul>
                <li>Enhance color separation</li>
                <li>Improve visual contrast</li>
                <li>Create more dynamic compositions</li>
                <li>Preserve artistic style while boosting vibrancy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)