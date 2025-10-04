# GVisMaps
Dataset of three sets of generalizable visibility maps for planning SAR observations in challenging terrain. Code for visualizing and processing generalizable visibility map data is included.
<img width="1700" height="1254" alt="nyc_capella" src="https://github.com/user-attachments/assets/4f2c9c47-09d3-4fbb-9c41-5c3bb227cda3" />

Generalizable visibility map data is stored in the folder "data." Use the script "visualize.py" to visualize the visibility map data and the script estimate_vis.py to estimate the visibility of a user-defined region of interest shape file.

If using this data or code in your own work, please reference the papers where the first description of SAR visibility maps were described:

E. L. Kramer and D. W. Miller, "Visibility Metric for Planning SAR Observations in Challenging Terrain," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-14, 2024, Art no. 5226114, doi: 10.1109/TGRS.2024.3487050.

Evan Kramer, David W Miller. Generalizable SAR visibility maps for planning observations in constrained mission scenarios. TechRxiv. September 03, 2025, doi: 10.36227/techrxiv.175693178.88101675/v1

Citations:
@ARTICLE{10736624,
  author={Kramer, Evan L. and Miller, David W.},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Visibility Metric for Planning SAR Observations in Challenging Terrain}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  keywords={Distortion;Geometry;Radar polarimetry;Satellite broadcasting;Satellites;Deformation;Terrain factors;Systematics;Software;Sentinel-1;Constellations;scheduling;synthetic aperture radar (SAR);visibility},
  doi={10.1109/TGRS.2024.3487050}}

@article{kramer2025generalizable,
  title={Generalizable SAR visibility maps for planning observations in constrained mission scenarios},
  author={Kramer, Evan and Miller, David W},
  year={2025},
  doi={10.36227/techrxiv.175693178.88101675/v1}
}  

Running the estimate_vis.py script will generate the estimate visibility map for the Manhattan road network case study presented in the Generalizable SAR visibility maps paper.
<img width="750" height="846" alt="Estimated_Visibility_Map_(Road-Oriented)" src="https://github.com/user-attachments/assets/47190887-ac12-42fd-a751-19fb62e62cf9" />
