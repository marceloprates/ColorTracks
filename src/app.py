import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import gpxpy
import io
import tempfile
import os
from src import (
    path2curve,
    process_and_smooth_path,
    standarize_path,
    compute_streamlines,
    load_track,
    main,
    compute_vector_field,
    draw_frame,
    stylize,
    fig2img,
)
from PIL import Image
from celluloid import Camera
import hashlib
import datetime

st.set_page_config(layout="wide")
# st.title("Static Print Generator")

# Add statistics fields
st.sidebar.header("Track Statistics")
distance = st.sidebar.number_input(
    "Distance (km)", min_value=0.0, value=0.0, step=0.1, format="%.1f"
)
avg_speed = st.sidebar.number_input(
    "Average Speed (km/h)", min_value=0.0, value=0.0, step=0.1, format="%.1f"
)
total_time = st.sidebar.text_input("Total Time", value="00:00:00")
kwh_consumption = st.sidebar.number_input(
    "kWh/100km", min_value=0.0, value=0.0, step=0.1, format="%.1f"
)
max_elevation = st.sidebar.number_input(
    "Maximum Elevation (m)", min_value=0.0, value=0.0, step=1.0, format="%.0f"
)
license_plate = st.sidebar.text_input("License Plate", value="", placeholder="BRA2E19")

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
user_name = st.sidebar.text_input("User Name", placeholder="Driver", value="")
event_date = st.sidebar.text_input("Date", value=current_date)

# Color palette selection
st.sidebar.header("Visual Parameters")
palette_choice = st.sidebar.radio(
    "Color Palette", ["Palette 1", "Palette 2"], help="Choose between two color schemes"
)

# Set palette based on selection
if palette_choice == "Palette 1":
    palette = ["#000", "#00A3E0", "#D0D3D4", "#73FDEA", "#BFDD0D"]
else:
    palette = ["#73FDEA", "#BFD000", "#E85B00", "#9B62F5"]

# Display the chosen palette
st.sidebar.write("Selected Palette:")
for color in palette:
    st.sidebar.markdown(
        f'<div style="background-color:{color};height:20px;border-radius:3px;margin:2px 0"></div>',
        unsafe_allow_html=True,
    )

# Background color selection
background_choice = st.sidebar.radio(
    "Background Color", ["Black", "White"], help="Choose the background color"
)

# Set background based on selection
background = "#000" if background_choice == "Black" else "#FFF"

# Display the chosen background
st.sidebar.write("Selected Background:")
st.sidebar.markdown(
    f'<div style="background-color:{background};height:20px;border-radius:3px;margin:2px 0"></div>',
    unsafe_allow_html=True,
)

# Animation vs Static selection
output_type = st.sidebar.radio(
    "Output Type",
    ["Static Image", "Animation"],
    help="Choose between a static image or animated visualization",
)

if output_type == "Animation":
    st.sidebar.header("Animation Parameters")
    frames = st.sidebar.slider("Number of Frames", 100, 1000, 400)
    fps = st.sidebar.slider("Frames per Second", 10, 60, 30)

# Image resolution settings
st.sidebar.header("Output Resolution")
resolution_preset = st.sidebar.radio(
    "Resolution Preset",
    ["16:9 HD", "16:9 4K", "Square", "A4 Portrait", "A4 Landscape", "Custom"],
    help="Choose from common resolution presets or set custom dimensions",
)

# Define preset resolutions (width, height, dpi)
resolution_presets = {
    "16:9 HD": (1920, 1080, 300),
    "16:9 4K": (3840, 2160, 300),
    "Square": (7020, 7020, 600),
    "A4 Portrait": (2480, 3508, 300),  # A4 at 300 DPI
    "A4 Landscape": (3508, 2480, 300),  # A4 at 300 DPI
}

if resolution_preset == "Custom":
    width = st.sidebar.number_input("Width (pixels)", min_value=100, value=1920)
    height = st.sidebar.number_input("Height (pixels)", min_value=100, value=1080)
    dpi = st.sidebar.number_input("DPI", min_value=72, value=300)
    figsize = (width / dpi, height / dpi)
else:
    width, height, dpi = resolution_presets[resolution_preset]
    figsize = (width / dpi, height / dpi)
    st.sidebar.write(f"Resolution: {width}x{height} pixels at {dpi} DPI")

# Stylization option
st.sidebar.header("Stylization")
apply_style = st.sidebar.radio(
    "Apply Artistic Style",
    ["No", "Yes"],
    help="Apply artistic stylization effects to the output",
)
apply_style = apply_style == "Yes"

# Add a slider from 0 to 1 called time
time_slider = st.slider("Time", 0.0, 1.0, 0.5, 0.01)


## Sidebar controls
# st.sidebar.header("Path Parameters")
# noise_level = st.sidebar.slider("Noise Level", 1e-10, 1e-8, 1e-9, format="%.10f")
# spline_smoothing = st.sidebar.slider(
#    "Spline Smoothing", 1e-7, 1e-5, 1e-6, format="%.7f"
# )
# spline_degree = st.sidebar.slider("Spline Degree", 2, 5, 3)
# num_points = st.sidebar.slider("Number of Points", 1000, 20000, 10000)
# sinusoidal_amplitude = st.sidebar.slider(
#    "Sinusoidal Amplitude", 1e-5, 1e-3, 1e-4, format="%.5f"
# )
# sinusoidal_frequency = st.sidebar.slider("Sinusoidal Frequency", 1, 20, 8)
#
# st.sidebar.header("Streamline Parameters")
# frames = st.sidebar.slider("Frames", 100, 1000, 400)
# num_deviations = st.sidebar.slider("Number of Deviations", 5, 50, 20)
# deviation_scale = st.sidebar.slider(
#    "Deviation Scale", 0.0001, 0.001, 0.0004, format="%.4f"
# )
# t_eval_points = st.sidebar.slider("Time Evaluation Points", 100, 1000, 500)

# Main content


st.header("Upload Path Data")
uploaded_file = st.file_uploader(
    "Choose a GPX file containing path coordinates", type=["gpx"]
)
generate_button = st.button("Generate Visualization")
if uploaded_file is not None and generate_button:
    # Load and process GPX data
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            gpx_path = tmp_file.name

        try:
            # Use session state to store computed values
            if "path" not in st.session_state or st.session_state.gpx_path != gpx_path:
                # Load the GPX track
                path = load_track(gpx_path)

                # Create a hash from the statistics
                hash_input = f"{distance}{avg_speed}{total_time}{kwh_consumption}{max_elevation}{license_plate}{user_name}{event_date}"
                stats_hash = hashlib.md5(hash_input.encode()).hexdigest()

                # Convert first 8 characters of hex hash to integer for seed
                seed_value = int(stats_hash[:8], 16)
                print(f"Stats hash: {stats_hash}")
                print(f"Generated seed: {seed_value}")
                st.info(f"Stats hash: {stats_hash[:8]} (Seed: {seed_value})")

                # Process and smooth the path
                print("Processing and smoothing the path...")
                x, y, t = process_and_smooth_path(path, show=False, seed=seed_value)

                # Compute vector field
                print("Computing the vector field...")
                (
                    t_func,
                    vector_field_for_plotting,
                    differential_equation_for_integration,
                ) = compute_vector_field(x, y, t, show=False, seed=seed_value)

                # Compute streamlines
                print("Computing streamlines...")
                streamlines = compute_streamlines(
                    x,
                    y,
                    t,
                    differential_equation_for_integration,
                    # frames=frames,
                    memoize=True,
                    show=False,
                )

                # Store values in session state
                st.session_state.gpx_path = gpx_path
                st.session_state.path = path
                st.session_state.x = x
                st.session_state.y = y
                st.session_state.t = t
                st.session_state.t_func = t_func
                st.session_state.vector_field_for_plotting = vector_field_for_plotting
                st.session_state.differential_equation_for_integration = (
                    differential_equation_for_integration
                )
                st.session_state.streamlines = streamlines
            else:
                # Use cached values from session state
                path = st.session_state.path
                x = st.session_state.x
                y = st.session_state.y
                t = st.session_state.t
                t_func = st.session_state.t_func
                vector_field_for_plotting = st.session_state.vector_field_for_plotting
                differential_equation_for_integration = (
                    st.session_state.differential_equation_for_integration
                )
                streamlines = st.session_state.streamlines

            if output_type == "Animation":
                print("Creating animation...")

                progress_bar = st.progress(0)

                # Create animation using celluloid
                fig, ax = plt.subplots(
                    figsize=(16, 9), constrained_layout=True, dpi=300
                )
                ax.clear()
                ax.axis("off")
                fig.patch.set_facecolor(background)
                camera = Camera(fig)

                # Create frames
                for i in range(frames):
                    t_val = (i / frames) * t.max()
                    draw_frame(fig, ax, t_val, t.max(), streamlines, palette)
                    camera.snap()

                    # Update progress
                    progress_bar.progress((i + 1) / frames)

                # Create and display animation
                animation = camera.animate()
                # Save animation as MP4
                # temp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                animation.save("tmp.mp4", writer="ffmpeg", fps=30)
                plt.close(fig)  # Close the figure before saving

                # Display video in Streamlit
                st.video(open("tmp.mp4", "rb").read())

                st.success("Animation complete!")

            else:
                # Draw a single frame
                print("Drawing a frame...")
                fig, ax = plt.subplots(
                    figsize=figsize, constrained_layout=True, dpi=dpi
                )
                ax.axis("off")
                fig.patch.set_facecolor(background)
                draw_frame(
                    fig, ax, time_slider * t.max(), t.max(), streamlines, palette
                )

                if apply_style:
                    img = fig2img(fig)
                    plt.close()
                    os.makedirs("prints", exist_ok=True)
                    img.save(
                        os.path.join(
                            "prints",
                            os.path.splitext(os.path.basename(gpx_path))[0] + ".png",
                        )
                    )

                    stylized_img = stylize(
                        img,
                        sigma=0,
                        bilateral_d=60,
                        bilateral_sigma_color=225,
                        bilateral_sigma_space=25,
                    )
                    # Convert the figure to a PIL image and display it in Streamlit
                    st.image(stylized_img, use_container_width=True)
                    plt.close()
                else:
                    st.pyplot(fig)
                    img = fig2img(fig)
                    os.makedirs("prints", exist_ok=True)
                    img.save("prints/a.png")

        finally:
            # Clean up the temporary file
            os.unlink(gpx_path)

    except Exception as e:
        st.error(f"Error processing path: {str(e)}")
