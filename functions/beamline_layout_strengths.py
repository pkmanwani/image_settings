import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import numpy as np
import re

# Constants
c = 299792458  # Speed of light in m/s

# Load JSON file
file_path = "../beamlines/awa/elements.json"  # Update with actual file path
with open(file_path) as file:
    beamline_data = json.load(file)

# Extract metadata (Beam energy and Lorentz factor)
pz = beamline_data["metadata"]["pz"] * 1e6  # Convert pz to eV
gamma = beamline_data["metadata"]["gamma"]
print(f"Beam Energy (pz): {pz} eV")
print(f"Lorentz Factor (γ): {gamma}")

# Extract beamline elements
quads = beamline_data["quads"]
yags = beamline_data["yags"]
drifts = beamline_data.get("drifts", [])  # Include drifts if present
ips = beamline_data.get("ips", [])  # Include IP elements if present

# Define fixed YAG screen length
yag_length = 0.1

# Convert positions to Elegant coordinates
reference_position = 10  # Adjust if needed
for quad in quads:
    quad["Elegant_Pos"] = quad["position"] - reference_position
for yag in yags:
    yag["Elegant_Pos"] = yag["position"] - reference_position
for drift in drifts:
    drift["Elegant_Pos"] = drift["position"] - reference_position
for ip in ips:
    ip["Elegant_Pos"] = ip["position"] - reference_position

# Read quadrupole strengths from the .lte file
quad_strengths = {}
lte_file = "save_lattice.filename.lte"  # Update with actual .lte file path
with open(lte_file, "r") as file:
    for line in file:
        match = re.match(r"(Q\d+): QUAD,L=.*?,K1=([\d\-.e]+)", line)
        if match:
            quad_name, k1_cm2 = match.groups()
            quad_strengths[quad_name] = float(k1_cm2) * pz / c  # Convert to T/m

# Plot settings
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(-0.5, max(q["Elegant_Pos"] for q in quads + yags + drifts + ips) + 1)  # Add some margin
ax.set_ylim(-1.2, 1.2)
ax.set_yticks([])  # Remove y-axis labels

# Draw quadrupoles (red rectangles with black borders)
for i, quad in enumerate(quads):
    rect = patches.Rectangle(
        (quad["Elegant_Pos"] - quad["length"] / 2, -0.3),
        quad["length"],
        0.6,
        facecolor="red",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    # Display quadrupole strength below
    k1_value = quad_strengths.get(quad["name"], 0)
    ax.text(
        quad["Elegant_Pos"], -0.7, f"{k1_value:.2f} ",
        ha="center", va="top", fontsize=10, color="blue", rotation=45
    )
    if i == 0:
        ax.text(
            quad["Elegant_Pos"] - 0.6, -0.7, "(T/m)",
            ha="center", va="top", fontsize=10, color="blue", rotation=45
        )

# Draw YAG screens (green rectangles with black borders)
for yag in yags:
    rect = patches.Rectangle(
        (yag["Elegant_Pos"] - yag_length / 2, -0.2),
        yag_length,
        0.4,
        facecolor="green",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(rect)

# Draw vertical dotted black lines for "IP" positions
for ip in ips:
    ax.axvline(x=ip["Elegant_Pos"], linestyle="dotted", color="black", linewidth=1.5)

# Label drifts (D* as text)
for drift in drifts:
    ax.text(
        drift["Elegant_Pos"] - drift["length"] / 2 + 0.05, 0.4, drift["name"],
        ha="center", va="bottom", fontsize=10, rotation=45
    )

# Label quadrupoles and YAG screens
for element in quads + yags:
    ax.text(
        element["Elegant_Pos"] + element["length"] / 2, -0.4, element["name"],
        ha="center", va="top", fontsize=10, rotation=45
    )

# Label IP positions
for ip in ips:
    ax.text(
        ip["Elegant_Pos"] - ip["length"] / 2 + 0.05, 0.6, ip["name"],
        ha="center", va="bottom", fontsize=10, rotation=45
    )

# Legend
handles = [
    patches.Patch(facecolor="red", edgecolor="black", linewidth=1.5, label="Quadrupole"),
    patches.Patch(facecolor="green", edgecolor="black", linewidth=1.5, label="YAG Screen"),
    plt.Line2D([0], [0], linestyle="dotted", color="black", linewidth=1.5, label="IP"),
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
ax.set_title("Beamline Layout with Quadrupole Strengths", fontsize=12)

plt.tight_layout()
plt.savefig("beamline_layout_strengths.png")
plt.show()
