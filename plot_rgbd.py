import matplotlib.pyplot as plt 
from datareader import *



parser = argparse.ArgumentParser()
# Configure logging
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, 
                    default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj'
                    # default=f"{code_dir}/demo_data/far_away3/mesh/model.obj"
                    )
parser.add_argument('--test_scene_dir', type=str, 
                    default=f"{code_dir}/data/mustard0"
                    )
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=3)
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
args = parser.parse_args()
reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

color=reader.get_color(100)
depth=reader.get_depth(100)
mask=reader.get_mask(100)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
cv2.imwrite('color.png',color)
mask=mask *255
mask=mask.astype(np.uint8)
cv2.imwrite('mask.png',mask)
# Normalize the depth data to the range [0, 255]
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply the "jet" colormap
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Save the colored depth map as an image
cv2.imwrite('depth.png', depth_colored)
# plt.figure()
# plt.imshow(color)
# plt.savefig('color.png')
# plt.close()
# plt.figure()
# plt.imshow(depth,cmap="jet")
# plt.savefig('depth.png')
# plt.close()
# plt.figure()
# plt.imshow(mask,cmap="gray")
# plt.savefig('mask.png')
# plt.close()
