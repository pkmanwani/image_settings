import json
import os
# Load JSON data
def generate_elegant_lattice(file_path):
    with open(file_path, "r") as file:
        beamline_data = json.load(file)

    # Define Elegant position reference (10m → Elegant position = 0)
    reference_position = 10  # Set 10m as zero

    # Extract elements
    quads = beamline_data["quads"]
    yags = beamline_data["yags"]

    # Convert positions to Elegant coordinates
    for quad in quads:
        quad["Elegant_Pos"] = quad["position"] - reference_position
    for yag in yags:
        if yag["position"] != "":
            yag["Elegant_Pos"] = yag["position"] - reference_position
        else:
            yag["Elegant_Pos"] = None  # Keep it None if position is missing

    # Prepare lattice elements
    lattice_elements = []
    yag_counter = 1

    # Generate lattice elements
    for quad in quads:
        if quad["Elegant_Pos"] > 0:
            lattice_elements.append(f"{quad['name']}: QUAD,L={quad['length']:.5f},K1=0")

    for yag in yags:
        if yag["Elegant_Pos"] >0:
            lattice_elements.append(f"W{yag['name'][3:]}: WATCH,filename=\"{yag['name']}.filename.sdds\"")
            yag_counter += 1

    # Create the LINE definition
    line_elements = [el.split(":")[0] for el in lattice_elements]
    lattice_content = "\n".join(lattice_elements) + f"\nL0001: LINE = ({','.join(line_elements)})\nBL1: LINE = (L0001)\nUSE,\"BL1\"\nRETURN\n"
    save_path = os.path.join(os.path.split(file_path)[0],"elegant.lte")
    # Save to Elegant lattice file
    with open(save_path, "w") as file:
        file.write(lattice_content)

    print("Lattice file 'elegant.lte' has been generated and saved.")


if __name__ == "__main__":
    file_path = "../beamlines/awa/elements.json"
    generate_elegant_lattice(file_path)
