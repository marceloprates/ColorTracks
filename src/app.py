import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
from matplotlib.animation import FuncAnimation
import tempfile
import os
from src import (
    process_and_smooth_path,
    compute_streamlines,
    load_track,
    main,
    compute_vector_field,
    draw_frame,
    fig2img,
    add_border,
)
from PIL import Image, ImageChops, ImageDraw, ImageFont
from celluloid import Camera
import hashlib
import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
FONT_PATH = "assets/PeugeotNew-Light.otf"
BORDER_PATH = "assets/2025_PEUGEOT_DesignWeek_Moldura copiar.png"
FONT_SIZE = 24


class ImageProcessor:
    @staticmethod
    def get_transparent_bbox(image):
        """Get the bounding box of the non-transparent area of an image."""
        alpha = image.split()[-1]
        inverted_alpha = ImageChops.invert(alpha)
        return inverted_alpha.getbbox()

    @staticmethod
    def add_text_to_border(border, user_name, event_date, font):
        """Add user name and event date text to the border image."""
        draw = ImageDraw.Draw(border)

        if user_name:
            draw.text(
                (0.25 * border.width, 0.19 * border.height),
                f"By {user_name}",
                fill="black",
                font=font,
            )

        if event_date:
            draw.text(
                (0.1 * border.width, 0.75 * border.height),
                f"{event_date}",
                fill="black",
                font=font,
            )

        return border


class VideoProcessor:
    @staticmethod
    def add_border_to_video(
        video_path: str,
        border_path: str,
        output_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
    ):
        """
        Overlays an MP4 animation within a specified bounding box inside a border image using ffmpeg.
        """
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            border_path,
            "-filter_complex",
            f"[0:v]scale={width}:{height}[scaled]; [1:v][scaled]overlay={x}:{y}",
            "-c:a",
            "copy",
            output_path,
            "-y",
        ]

        subprocess.run(command, check=True)

    @staticmethod
    def repeat_video(input_path, output_path, repeats=4):
        """Repeat a video multiple times."""
        subprocess.run(
            [
                "ffmpeg",
                "-stream_loop",
                str(repeats),
                "-i",
                input_path,
                "-c",
                "copy",
                output_path,
                "-y",
            ]
        )


class GPXVisualizer:
    def __init__(self):
        self.load_font_and_border()
        self.setup_page_config()

    def load_font_and_border(self):
        """Initialize font and border assets."""
        self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        self.border = Image.open(BORDER_PATH)

    def setup_page_config(self):
        """Configure Streamlit page layout."""
        st.set_page_config(layout="wide")

    def setup_sidebar(self):
        """Configure sidebar inputs and parameters."""
        self._setup_track_statistics()
        self._setup_visual_parameters()
        self._setup_output_options()
        self._setup_border_options()

    def _setup_track_statistics(self):
        """Setup track statistics inputs."""
        st.sidebar.header("Track Statistics")
        self.distance = st.sidebar.number_input(
            "Distance (km)", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
        self.avg_speed = st.sidebar.number_input(
            "Average Speed (km/h)", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
        self.total_time = st.sidebar.text_input("Total Time", value="00:00:00")
        self.kwh_consumption = st.sidebar.number_input(
            "kWh/100km", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
        self.max_elevation = st.sidebar.number_input(
            "Maximum Elevation (m)", min_value=0.0, value=0.0, step=1.0, format="%.0f"
        )
        self.license_plate = st.sidebar.text_input(
            "License Plate", value="", placeholder="BRA2E19"
        )

        current_date = datetime.datetime.now().strftime("%d/%m/%Y")
        self.user_name = st.sidebar.text_input(
            "User Name", placeholder="Driver", value=""
        )
        self.event_date = st.sidebar.text_input("Date", value=current_date)

    def _setup_visual_parameters(self):
        """Setup visual parameters inputs."""
        st.sidebar.header("Visual Parameters")
        palette_choice = st.sidebar.radio(
            "Color Palette",
            ["Palette 1", "Palette 2"],
            help="Choose between two color schemes",
        )

        self.route_selection = st.sidebar.radio(
            "Select Route",
            ["Jardim Paulista 1", "Jardim Paulista 2", "Pinheiros 1", "Pinheiros 2"],
            help="Choose from predefined routes or upload your own",
        )
        route_files = {
            "Jardim Paulista 1": "assets/paths/jardim-paulista_1.gpx",
            "Jardim Paulista 2": "assets/paths/jardim-paulista_2.gpx",
            "Pinheiros 1": "assets/paths/pinheiros_1.gpx",
            "Pinheiros 2": "assets/paths/pinheiros_2.gpx",
        }
        self.gpx_path = route_files[self.route_selection]
        self.uploaded_file = None
        self.use_uploaded_file = False

        # Set palette based on selection
        if palette_choice == "Palette 1":
            self.palette = ["#000", "#00A3E0", "#D0D3D4", "#73FDEA", "#BFDD0D"]
        else:
            self.palette = ["#73FDEA", "#BFD000", "#E85B00", "#9B62F5"]

        # Display the chosen palette
        st.sidebar.write("Selected Palette:")
        for color in self.palette:
            st.sidebar.markdown(
                f'<div style="background-color:{color};height:20px;border-radius:3px;margin:2px 0"></div>',
                unsafe_allow_html=True,
            )

        # Background color selection
        background_choice = st.sidebar.radio(
            "Background Color", ["White", "Black"], help="Choose the background color"
        )

        # Set background based on selection
        self.background = "#000" if background_choice == "Black" else "#FFF"

        # Display the chosen background
        st.sidebar.write("Selected Background:")
        st.sidebar.markdown(
            f'<div style="background-color:{self.background};height:20px;border-radius:3px;margin:2px 0"></div>',
            unsafe_allow_html=True,
        )

    def _setup_output_options(self):
        """Setup output type options."""
        # Animation vs Static selection
        self.output_type = st.sidebar.radio(
            "Output Type",
            ["Static Image", "Animation"],
            help="Choose between a static image or animated visualization",
        )

        if self.output_type == "Animation":
            st.sidebar.header("Animation Parameters")
            self.frames = st.sidebar.slider("Number of Frames", 100, 1000, 450)
            self.fps = st.sidebar.slider("Frames per Second", 15, 60, 30, 15)

    def _setup_border_options(self):
        """Setup border options."""
        st.sidebar.header("Border Options")
        border_option = st.sidebar.radio(
            "Display Border",
            ["With Border", "Without Border"],
            help="Choose whether to display the image with or without the Peugeot border",
        )
        self.show_border = border_option == "With Border"

        # Set figure dimensions based on border choice
        self.dpi = 150
        if self.show_border:
            self.figsize = (878 / self.dpi, 878 / self.dpi)
        else:
            self.figsize = (1920 / self.dpi, 1080 / self.dpi)

    def setup_main_content(self):
        """Setup main content area with file uploader and time slider."""
        self.time_slider = st.slider("Time", 0.0, 1.0, 0.5, 0.01)

        self.generate_button = st.button("Generate Visualization")

    def process_gpx_data(self):
        """Process uploaded GPX data and generate visualization."""
        if not self.generate_button:
            return

        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp_file:
                # Read from the predefined route path
                with open(self.gpx_path, "rb") as src_file:
                    tmp_file.write(src_file.read())
                tmp_file.flush()
                gpx_path = tmp_file.name
                print(gpx_path)

            try:
                self._load_or_compute_path_data(gpx_path)

                if self.output_type == "Animation":
                    self._generate_animation()
                else:
                    self._generate_static_image()

            finally:
                # Clean up the temporary file
                os.unlink(gpx_path)
                # st.runtime.legacy_caching.clear_cache()

        except Exception as e:
            st.error(f"Error processing path: {str(e)}")

    def _load_or_compute_path_data(self, gpx_path):
        """Load or compute path data from GPX file."""
        # Use session state to store computed values
        if "path" not in st.session_state or st.session_state.gpx_path != gpx_path:
            # Load the GPX track
            path = load_track(gpx_path)

            # Create a hash from the statistics
            hash_input = f"{self.distance}{self.avg_speed}{self.total_time}{self.kwh_consumption}{self.max_elevation}{self.license_plate}{self.user_name}{self.event_date}"
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

        # Set instance variables from session state
        self.path = st.session_state.path
        self.x = st.session_state.x
        self.y = st.session_state.y
        self.t = st.session_state.t
        self.t_func = st.session_state.t_func
        self.vector_field_for_plotting = st.session_state.vector_field_for_plotting
        self.differential_equation_for_integration = (
            st.session_state.differential_equation_for_integration
        )
        self.streamlines = st.session_state.streamlines

    def _generate_animation(self):
        """Generate and display animation."""
        print("Creating animation...")
        progress_bar = st.progress(0)

        fig, ax = plt.subplots(
            figsize=self.figsize, constrained_layout=True, dpi=self.dpi
        )
        ax.axis("off")
        fig.patch.set_facecolor(self.background)

        # Determine the fixed axes limits based on the data
        x_min, x_max = min(self.x), max(self.x)
        y_min, y_max = min(self.y), max(self.y)
        # Add a small margin
        margin_x = 0.1 * (x_max - x_min)
        margin_y = 0.1 * (y_max - y_min)

        # Create a progress bar that can be updated from within the animation function
        progress_bar_placeholder = st.empty()
        progress_bar = progress_bar_placeholder.progress(0)

        # Create update function for FuncAnimation
        def update(frame):
            ax.clear()
            ax.axis("off")
            # Set fixed limits
            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_min - margin_y, y_max + margin_y)

            t_val = (frame / self.frames) * self.t.max()
            draw_frame(
                fig,
                ax,
                t_val,
                self.t.max(),
                self.streamlines,
                self.palette,
            )
            # Update progress bar
            progress_bar.progress(frame / self.frames)
            return ax.get_children()

        # Create animation
        anim = FuncAnimation(fig, update, frames=self.frames, blit=True)

        # Save animation as MP4
        original_filename = os.path.splitext(os.path.basename(self.gpx_path))[0]
        os.makedirs("animations", exist_ok=True)
        output_filename = f"animations/{original_filename}.mp4"

        # Use a lower DPI for the writer to reduce memory usage
        writer = plt.matplotlib.animation.FFMpegWriter(
            fps=self.fps, metadata=dict(artist="GPXVisualizer"), bitrate=1800
        )

        anim.save(output_filename, writer=writer)
        plt.close(fig)  # Explicitly close the figure to free memory

        # Repeat the animation
        repeated_filename = f"animations/{original_filename}_repeated.mp4"
        bordered_filename = f"animations/{original_filename}_repeated_bordered.mp4"
        VideoProcessor.repeat_video(output_filename, repeated_filename)

        final_video_path = repeated_filename

        # Add border if needed
        if self.show_border:
            border_img = ImageProcessor.add_text_to_border(
                self.border.copy(),
                user_name=self.user_name,
                event_date=self.event_date,
                font=self.font,
            )
            # Resize 25%
            border_img = border_img.resize(
                (int(border_img.width / 2), int(border_img.height / 2))
            )
            border_img.save("tmp.png")

            bbox = ImageProcessor.get_transparent_bbox(border_img)
            x, y, x2, y2 = bbox
            width, height = x2 - x, y2 - y

            VideoProcessor.add_border_to_video(
                video_path=repeated_filename,
                border_path="tmp.png",
                output_path=bordered_filename,
                x=x,
                y=y,
                width=width,
                height=height,
            )
            final_video_path = bordered_filename

        # Display the final video
        st.video(final_video_path)
        # Add download button for video
        with open(final_video_path, "rb") as file:
            btn = st.download_button(
                label="Download Video",
                data=file,
                file_name=f"{original_filename}.mp4",
                mime="video/mp4",
            )
        st.success("Animation complete!")

    def _generate_static_image(self):
        """Generate and display static image."""
        print("Drawing a frame...")
        fig, ax = plt.subplots(
            figsize=self.figsize, constrained_layout=True, dpi=self.dpi
        )
        ax.axis("off")
        fig.patch.set_facecolor(self.background)

        draw_frame(
            fig,
            ax,
            self.time_slider * self.t.max(),
            self.t.max(),
            self.streamlines,
            self.palette,
        )

        img = fig2img(fig)
        # Save to temp file
        img.save("tmp.png")

        if self.show_border:
            border_img = ImageProcessor.add_text_to_border(
                self.border.copy(),
                user_name=self.user_name,
                event_date=self.event_date,
                font=self.font,
            )
            img = add_border(
                img,
                border_img,
                user_name=self.user_name,
                event_date=self.event_date,
                font=self.font,
            )
            st.image(img, use_container_width=True)
        else:
            st.pyplot(fig)

        # Add download button
        original_filename = os.path.splitext(self.gpx_path)[1]
        output_path = f"{original_filename}.png"
        img.save(output_path)

        # Create a download button using Streamlit's built-in function
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name=f"{original_filename}.png",
                mime="image/png",
            )


def main():
    visualizer = GPXVisualizer()
    visualizer.setup_sidebar()
    visualizer.setup_main_content()
    visualizer.process_gpx_data()


if __name__ == "__main__":
    main()
