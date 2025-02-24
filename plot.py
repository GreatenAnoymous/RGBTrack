import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX-style fonts


plt.rcParams.update({
    "text.usetex": False,  # Disable full LaTeX rendering
    "font.family": "serif",  # Use a built-in serif font
    "mathtext.fontset": "cm"  # Use Computer Modern for math text
})


# Read CSV files
data = pd.read_csv("foundation_pose_traj.csv")  
data1 = pd.read_csv("tracking_traj.csv")  
data2 = pd.read_csv("tracking_no_depth_traj.csv")

# Generate step indices as x-axis
data["step"] = range(1, len(data) + 1)
data1["step"] = range(1, len(data1) + 1)
data2["step"] = range(1, len(data2) + 1)

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

# Plot Recall
axes[0].plot(data["step"], data["recall"],  linestyle="--", label=r"FoundationPose (RGBD)")
axes[0].plot(data1["step"], data1["recall"], linestyle="--", label=r"Ours (RGBD)")
axes[0].plot(data2["step"], data2["recall"], linestyle="--", label=r"Ours (RGB)")
axes[0].set_ylabel(r"Recall")
axes[0].set_title(r"Tracking Performance")
# axes[0].legend()
# axes[0].grid()

# Plot Translation Error
axes[1].plot(data["step"], data["translation_error"],  linestyle="--", label=r"FoundationPose (RGBD)")
axes[1].plot(data1["step"], data1["translation_error"],  linestyle="--", label=r"Ours (RGBD)")
axes[1].plot(data2["step"], data2["translation_error"],  linestyle="--", label=r"Ours (RGB)")
axes[1].set_ylabel(r"Translation Error (m)")
axes[1].legend()
# axes[1].grid()

# Plot Rotation Error
axes[2].plot(data["step"], data["rotation_error_deg"], linestyle="--", label=r"FoundationPose (RGBD)")
axes[2].plot(data1["step"], data1["rotation_error_deg"],  linestyle="--", label=r"Ours (RGBD)")
axes[2].plot(data2["step"], data2["rotation_error_deg"],  linestyle="--", label=r"Ours (RGB)")
axes[2].set_xlabel(r"Time Step")
axes[2].set_ylabel(r"Rotation Error (deg)")
# axes[2].legend()
# axes[2].grid()

# Adjust layout and save figure
plt.tight_layout()
# plt.savefig("./tracking_pic.png", dpi=300)
plt.savefig("./tracking_pic.pdf",bbox_inches='tight')
plt.show()
