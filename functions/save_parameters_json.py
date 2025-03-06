import sys
sys.path.append('../')
from image_fit import image_fit
import os
import json

def save_parameters_to_json(folder_path,roi=True,sigma_size = 3,get_res=False,mask_every_image=False,debug=False,calc_jitter=False):
    results = []

    # Loop through each file in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
            # Call the image_fit function
            output = image_fit(file_path, roi, sigma_size, get_res, mask_every_image, debug, calc_jitter)
            print(output)
            if output is None:  # If image_fit returns None, store None in results
                print("if")
                result = {
                    'filename': filename,
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
            else:
                print(output)
                print("else")
                parameters, errors = output
                Cx_final, Cy_final, Sx_final, Sy_final, Sxy_final, Rx_final, Ry_final, Rxy_final, angle, res = parameters
                Cx_final_e, Cy_final_e, Sx_final_e, Sy_final_e, Sxy_final_e, Rx_final_e, Ry_final_e, Rxy_final_e, angle_e, res_e = errors

                # Create a result dictionary
                result = {
                    'filename': filename,
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

            results.append(result)

    # Define the JSON file name based on the folder name
    json_file_path = os.path.join(folder_path, 'parameters.json')

    # Save the results to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Saved parameters to {json_file_path}')

if __name__ == "__main__":
    # Example usage
    folder_path = '../beamlines/awa/2025_02_19-selected/3ScreenMeasurement_LongPulse'  # Change this to your folder path
    save_parameters_to_json(folder_path,roi=False,calc_jitter=True)
