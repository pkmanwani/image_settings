import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import os
def plot_beamline(file_path):
    with open(file_path) as file:
        beamline_data = json.load(file)
    # User-defined Elegant position reference (set 10m to Elegant_pos = 0)
    reference_position = 10  # Set reference position at 10m

    # Extract elements
    quads = beamline_data["quads"]
    yags = beamline_data["yags"]

    # Define fixed YAG screen length
    yag_length = 0.1

    # Convert positions to Elegant_pos
    for quad in quads:
        quad["Elegant_Pos"] = quad["position"] - reference_position
    for yag in yags:
        if yag["position"] != "":
            yag["Elegant_Pos"] = yag["position"] - reference_position
        else:
            yag["Elegant_Pos"] = None  # Keep it None if no position is provided

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(-0.5, max(q["Elegant_Pos"] for q in quads) + 1)  # Add some margin
    ax.set_ylim(-1, 1)
    ax.set_yticks([])  # Remove y-axis labels

    # Draw quadrupoles (red rectangles with black borders)
    for quad in quads:
        if quad["Elegant_Pos"] > 0:
            rect = patches.Rectangle(
                (quad["Elegant_Pos"] - quad["length"] / 2, -0.3),
                quad["length"],
                0.6,
                facecolor="red",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                quad["Elegant_Pos"] + quad["length"] / 2,
                -0.4,
                quad["name"],
                ha="center",
                va="top",
                fontsize=10,
                rotation=45,
            )

    # Draw YAG screens (green rectangles with black borders)
    for yag in yags:
        if yag["Elegant_Pos"] > 0:
            rect = patches.Rectangle(
                (yag["Elegant_Pos"] - yag_length / 2, -0.2),
                yag_length,
                0.4,
                facecolor="green",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                yag["Elegant_Pos"] + yag_length / 2,
                -0.4,
                yag["name"],
                ha="center",
                va="top",
                fontsize=10,
                rotation=45,
            )

    # Legend
    handles = [
        patches.Patch(facecolor="red", edgecolor="black", linewidth=1.5, label="Quadrupole"),
        patches.Patch(facecolor="green", edgecolor="black", linewidth=1.5, label="YAG Screen"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10)

    # Create a secondary x-axis (starting from 10 m)
    def secondary_x(x):
        return x + reference_position  # Shift by 10 m


    secax = ax.secondary_xaxis("bottom", functions=(secondary_x, lambda x: x - reference_position))
    secax.xaxis.set_label_position("bottom")
    secax.xaxis.set_ticks_position("bottom")
    secax.spines["bottom"].set_position(("outward", 20))  # Move it slightly below

    # Set tick marks for both axes
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    secax.xaxis.set_major_locator(MultipleLocator(1))
    secax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # Labels
    secax.set_xlabel("AWA Position (m)", fontsize=12)
    ax.set_title("Beamline Layout", fontsize=12)
    #plt.xlim(left=reference_position)
    plt.tight_layout()
    #print(os.path.split(file_path)[0])
    plt.savefig(os.path.join(os.path.split(file_path)[0],"beamline_layout.png"))
    plt.show()

if __name__ == "__main__":
    file_path = "../beamlines/awa/elements.json"
    plot_beamline(file_path)
