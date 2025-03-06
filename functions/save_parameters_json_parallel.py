import sys
sys.path.append('../')
from image_fit import image_fit
import os
import json
from concurrent.futures import ProcessPoolExecutor


def process_file(file_path, roi, sigma_size, get_res, mask_every_image, debug, calc_jitter):
    """ Function to process a single file using image_fit and return results """
    output = image_fit(file_path, roi, sigma_size, get_res, mask_every_image, debug, calc_jitter)

    if output is None:
        return {
            'filename': os.path.basename(file_path),
            'Cx_final': None,
            'Cy_final': None,
            'Sx_final': None,
            'Sy_final': None,
            'Sxy_final': None,
            'Rx_final': None,
            'Ry_final': None,
            'Rxy_final': None,
            'angle': None,
            'res': None
        }
    parameters, errors = output
    Cx_final, Cy_final, Sx_final, Sy_final, Sxy_final, Rx_final, Ry_final, Rxy_final, angle, res = parameters
    Cx_final_e, Cy_final_e, Sx_final_e, Sy_final_e, Sxy_final_e, Rx_final_e, Ry_final_e, Rxy_final_e, angle_e, res_e = errors

    return {
        'filename': os.path.basename(file_path),
        'Cx_final': [Cx_final, Cx_final_e],
        'Cy_final': [Cy_final, Cy_final_e],
        'Sx_final': [Sx_final, Sx_final_e],
        'Sy_final': [Sy_final, Sy_final_e],
        'Sxy_final': [Sxy_final, Sxy_final_e],
        'Rx_final': [Rx_final, Rx_final_e],
        'Ry_final': [Ry_final, Ry_final_e],
        'Rxy_final': [Rxy_final, Rxy_final_e],
        'angle': [angle, angle_e],
        'res': [res, res_e]
    }


def save_parameters_to_json_parallel(folder_path, roi=True, sigma_size=3, get_res=False, mask_every_image=False, debug=False,
                            calc_jitter=False):
    """ Function to process all files in parallel and save results to JSON """
    h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
    results = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, file_path, roi, sigma_size, get_res, mask_every_image, debug, calc_jitter) for
            file_path in h5_files]

        for future in futures:
            results.append(future.result())

    # Save results to JSON file
    json_file_path = os.path.join(folder_path, 'parameters.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Saved parameters to {json_file_path}')

if __name__ == "__main__":
    # Example usage
    folder_path = '../beamlines/awa/2025_02_19-selected/3ScreenMeasurement_LongPulse'  # Change this to your folder path
    save_parameters_to_json_parallel(folder_path,roi=False,calc_jitter=True)