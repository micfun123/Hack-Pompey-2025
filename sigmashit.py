
def get_pasquill_stability(wind_speed, daytime=True, solar_radiation='moderate', cloud_cover=0.5):
    """
    Determines Pasquill-Gifford stability class.
    
    Args:
        wind_speed (float): Wind speed (m/s)
        daytime (bool): True if daytime, False if nighttime
        solar_radiation (str): 'strong', 'moderate', 'slight' (daytime)
        cloud_cover (float): 0-1 (0=clear, 1=overcast) (nighttime)
    
    Returns:
        str: Stability class (A-F)
    """
    if daytime:
        if wind_speed < 2:
            return 'A' if solar_radiation == 'strong' else 'B'
        elif wind_speed < 3:
            return 'B'
        elif wind_speed < 5:
            return 'C' if solar_radiation == 'slight' else 'B'
        else:
            return 'D'
    else:  # Nighttime
        if wind_speed < 2:
            return 'F' if cloud_cover < 0.4 else 'E'
        elif wind_speed < 5:
            return 'E' if cloud_cover < 0.4 else 'D'
        else:
            return 'D'

def compute_sigma_y_sigma_z(stability_class, x):
    """
    Computes sigma_y and sigma_z using Briggs rural formulas.
    
    Args:
        stability_class (str): 'A'-'F'
        x (float): Downwind distance (m)
    
    Returns:
        tuple: (sigma_y, sigma_z) in meters
    """
    stability_to_sigma = {
        'A': (0.22 * x / np.sqrt(1 + 0.0001*x), 0.20 * x),
        'B': (0.16 * x / np.sqrt(1 + 0.0001*x), 0.12 * x),
        'C': (0.11 * x / np.sqrt(1 + 0.0001*x), 0.08 * x / np.sqrt(1 + 0.0002*x)),
        'D': (0.08 * x / np.sqrt(1 + 0.0001*x), 0.06 * x / np.sqrt(1 + 0.0015*x)),
        'E': (0.06 * x / np.sqrt(1 + 0.0001*x), 0.03 * x / (1 + 0.0003*x)),
        'F': (0.04 * x / np.sqrt(1 + 0.0001*x), 0.016 * x / (1 + 0.0003*x))
    }
    return stability_to_sigma[stability_class]
