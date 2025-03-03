import streamlit as st
from utils.style_generator import StyleTransfer
from utils.ui_components import (
    setup_page_config,
    apply_custom_css,
    render_header,
    render_controls,
    render_image_columns,
    render_example_gallery,
    render_info_sections
)

# Initialize the application
setup_page_config()
apply_custom_css()
render_header()

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = StyleTransfer.get_instance()
    if not st.session_state.generator.is_initialized:
        st.session_state.generator.initialize_pipeline()

# Render controls and handle user input
prompt, selected_style = render_controls(st.session_state.generator.style_names)

if st.sidebar.button("🚀 Generate Artwork", use_container_width=True):
    if prompt:
        try:
            with st.spinner("Generating your artwork..."):
                base_image, enhanced_image = st.session_state.generator.generate_artwork(prompt, selected_style)
                
                # Store images in session state
                st.session_state.base_image = base_image
                st.session_state.enhanced_image = enhanced_image
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a prompt first!")

# Display generated images
render_image_columns(
    base_image=st.session_state.get('base_image'),
    enhanced_image=st.session_state.get('enhanced_image')
)

# Render example gallery and information sections
render_example_gallery()
render_info_sections()