import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pandas as pd
from flask import Flask, Response, render_template, url_for
import io
import os
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap

app = Flask(__name__)

# Configuration
PM_TYPE = "PM25"                  # Default PM type
DEFAULT_DAY = 0                   # Default day index
GRID_SIZE = 100                   # Plume grid resolution
PLUME_RANGE = 100000           # Increased to extend the plume coverage
PLUME_ALPHA = 0.85                # Plume opacity
BASE_IMAGE_PATH = "static/map_background.png"  # Local base image path


def get_portsmouth_sigmas(x, stability_class='D', wind_speed=1.0):
    """More physically accurate dispersion coefficients"""
    stability_params = {
        'A': (0.32, 0.24, 0.0004),  
        'B': (0.22, 0.20, 0.0004),
        'C': (0.16, 0.14, 0.0003),
        'D': (0.11, 0.08, 0.0003),  
        'E': (0.08, 0.06, 0.0003),
        'F': (0.06, 0.03, 0.0003)
    }
    
    ay, az, bx = stability_params.get(stability_class, (0.11, 0.08, 0.0003))
    wind_factor = 1.5 + (wind_speed / 5.0)  # Increased scaling for larger plume
    ay *= wind_factor
    az *= wind_factor * 0.8  
    
    sigma_y = ay * x * (1 + bx * x)**-0.5
    sigma_z = az * x * (1 + bx * x)**-0.5
    
    return sigma_y, sigma_z

def gaussian_plume(x_grid, y_grid, Q, wind_speed, wind_dir, stability_class='D'):
    if wind_speed <= 0.1:
        return np.zeros_like(x_grid)
    
    theta = np.radians(wind_dir)
    x_rot = x_grid * np.cos(theta) + y_grid * np.sin(theta)
    y_rot = -x_grid * np.sin(theta) + y_grid * np.cos(theta)
    
    C = np.zeros_like(x_grid)
    downwind = x_rot > 0
    max_dist = PLUME_RANGE
    x_scaled = x_rot[downwind] / max_dist
    y_scaled = y_rot[downwind] / max_dist
    
    sigma_y, sigma_z = get_portsmouth_sigmas(x_scaled, stability_class, wind_speed)
    C[downwind] = (Q / (2 * np.pi * wind_speed * sigma_y * sigma_z)) * \
                  np.exp(-0.5 * (y_scaled/sigma_y)**2) * \
                  np.exp(-0.5 * (0/sigma_z)**2)
    
    scale_factor = 500.0 / (max_dist ** 2)  # Further increased scaling to extend plume further
    C[downwind] *= scale_factor * 10000
    C = (C / (np.max(C) + 1e-9)) ** 0.18  # Adjusted contrast exponent for even larger plume
    
    return C

def create_plume_cmap():
    """Create colormap with custom transparency."""
    hot = plt.cm.get_cmap('hot', 256)
    new_colors = hot(np.linspace(0, 1, 256))
    new_colors[:, -1] = np.linspace(0, PLUME_ALPHA, 256)  # Alpha ramp
    new_colors[:64, -1] = 0  # Lowest 25% transparent
    return LinearSegmentedColormap.from_list('transparent_hot', new_colors)

def load_base_image():
    """Load local base image with error handling."""
    try:
        return Image.open(BASE_IMAGE_PATH).convert('RGBA')
    except FileNotFoundError:
        # Create a blank image if base image not found
        print(f"Base image not found at {BASE_IMAGE_PATH}, using blank image")
        return Image.new('RGBA', (600, 600), (255, 255, 255, 255))

def create_plume_overlay(plume_data):
    """Create transparent plume overlay image."""
    fig = plt.figure(figsize=(6, 6), dpi=100, facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
    ax.set_axis_off()
    
    cmap = create_plume_cmap()
    ax.imshow(
        plume_data,
        cmap=cmap,
        interpolation='bilinear',
        origin='lower',
        extent=[-PLUME_RANGE, PLUME_RANGE, -PLUME_RANGE, PLUME_RANGE],
        vmin=0.1,
        vmax=1.0
    )
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def combine_images(base_img, overlay_img):
    """Combine base image with plume overlay."""
    # Resize overlay to match base image dimensions
    overlay_img = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
    
    # Ensure both images have alpha channel
    if base_img.mode != 'RGBA':
        base_img = base_img.convert('RGBA')
    if overlay_img.mode != 'RGBA':
        overlay_img = overlay_img.convert('RGBA')
    
    return Image.alpha_composite(base_img, overlay_img)


@app.route('/plume_overlay')
@app.route('/plume_overlay/<int:day>')
@app.route('/plume_overlay/<int:day>/<pm_type>')
def generate_plume_overlay(day=DEFAULT_DAY, pm_type=PM_TYPE):
    pm_type = pm_type.upper()
    if pm_type not in ["PM1", "PM10", "PM25"]:
        return "Invalid PM type", 400
    
    try:
        with open("output.json") as f:
            data = json.load(f)
        df = pd.DataFrame([
            (date, float(v["Wind Speed"]), float(v["Wind Dir"]), 
             float(v["PM1"]), float(v["PM10"]), float(v["PM25"]))
            for date, v in data.items()
        ], columns=["date", "Wind Speed", "Wind Dir", "PM1", "PM10", "PM25"])
        entry = df.iloc[day]
    except Exception as e:
        return f"Data error: {str(e)}", 400
    
    x = y = np.linspace(-PLUME_RANGE, PLUME_RANGE, GRID_SIZE)
    x_grid, y_grid = np.meshgrid(x, y)
    
    plume = gaussian_plume(
        x_grid, y_grid,
        Q=entry[pm_type] * 10000,
        wind_speed=entry["Wind Speed"],
        wind_dir=entry["Wind Dir"],
        stability_class='D'
    )
    
    base_img = load_base_image()
    overlay_img = create_plume_overlay(plume)
    final_image = combine_images(base_img, overlay_img)
    
    img_buffer = io.BytesIO()
    final_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return Response(
        img_buffer.getvalue(),
        mimetype='image/png',
        headers={
            'Cache-Control': 'no-cache',
            'Content-Disposition': 'inline; filename="plume_overlay.png"'
        }
    )




@app.route('/')
def index():
    try:
        with open("output.json") as f:
            data = json.load(f)
        df = pd.DataFrame([
            (date, float(v["Wind Speed"]), float(v["Wind Dir"]))
            for date, v in data.items()
        ], columns=["date", "Wind Speed", "Wind Dir"])
        entry = df.iloc[DEFAULT_DAY]
    except Exception:
        entry = {"Wind Speed": "N/A", "Wind Dir": "N/A"}
    
    return render_template('index.html', 
                           default_day=DEFAULT_DAY,
                           initial_image=url_for('generate_plume_overlay', day=DEFAULT_DAY, pm_type=PM_TYPE),
                           wind_speed=entry["Wind Speed"],
                           wind_dir=entry["Wind Dir"])

@app.route('/wind_data/<int:day>')
def get_wind_data(day):
    try:
        with open("output.json") as f:
            data = json.load(f)
        df = pd.DataFrame([
            (date, float(v["Wind Speed"]), float(v["Wind Dir"]))
            for date, v in data.items()
        ], columns=["date", "Wind Speed", "Wind Dir"])
        entry = df.iloc[day]
    except Exception:
        return {"wind_speed": "N/A", "wind_dir": "N/A"}, 400

    return {"wind_speed": entry["Wind Speed"], "wind_dir": entry["Wind Dir"]}


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=6005, debug=True)
