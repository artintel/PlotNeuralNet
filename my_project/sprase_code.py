
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    # to_input( '../examples/fcn8s/cats.jpg' ),


    # face-pipeline
    # to_Conv("face_conv1", 256, 1, width=1, height=20, depth=20),
    to_SoftMax("face_feature_vector", 200),
    to_Gallery("gallery", offset="(3,0,0)", to="(face_feature_vector-east)", height=26, depth=26, width=13, caption="faces-gallery", opacity=0.5),
    to_Gallery_map("gallery_map_m1", offset="(-2.3,0,0)", to="(gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("gallery_map_m2", offset="(-1.5,0,0)", to="(gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("gallery_map_m3", offset="(-0.8,0,0)", to="(gallery-east)", height=24, depth=24, width=2.5 ),
    to_Conv("face_sprase_coding", 1, 200, offset="(3,0,0)", to="(gallery-east)", width=25, height=3, depth=1.5, caption="face-sprase-coding"),
    to_Sum("face_sum1", offset="(5,0,0)", to="(face_sprase_coding-east)", radius=0, opacity=0.0),
    to_Sum("face_sum2", offset="(0,-9.0,0)",to="(face_sum1-south)", radius=0, opacity=0.0),
    
    
    # # ear-pipeline
    to_SoftMax("ear_feature_vector", 200, offset="(0,-20,0)", to="(0,0,0)"),
    to_Gallery("ear_gallery", offset="(3,0,0)", to="(ear_feature_vector-east)", height=26, depth=26, width=13, caption="faces-gallery", opacity=0.5),
    to_Gallery_map("ear_gallery_map_m1", offset="(-2.3,0,0)", to="(ear_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("ear_gallery_map_m2", offset="(-1.5,0,0)", to="(ear_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("ear_gallery_map_m3", offset="(-0.8,0,0)", to="(ear_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Conv("ear_sprase_coding", 1, 200, offset="(3,0,0)", to="(ear_gallery-east)", width=25, height=3, depth=1.5, caption="ear-sprase-coding"),
    to_Sum("ear_sum1", offset="(5,0,0)", to="(ear_sprase_coding-east)", radius=0, opacity=0.0),
    to_Sum("ear_sum2", offset="(0,8.6,0)",to="(ear_sum1-south)", radius=0, opacity=0.0),

    # # fusion-pipeline
    to_SoftMax("fusion_feature_vector", 200, offset="(0,-10,0)", to="(0,0,0)"),
    to_Gallery("fusion_gallery", offset="(3,0,0)", to="(fusion_feature_vector-east)", height=26, depth=26, width=13, caption="fusion-gallery", opacity=0.5),
    to_Gallery_map("fusion_gallery_map_m1", offset="(-2.3,0,0)", to="(fusion_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("fusion_gallery_map_m2", offset="(-1.5,0,0)", to="(fusion_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("fusion_gallery_map_m3", offset="(-0.8,0,0)", to="(fusion_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Conv("joint_sprase_coding", 1, 200, offset="(3,0,0)", to="(fusion_gallery-east)", width=25, height=3, depth=1.5, caption="joint-sprase-coding"),
    to_Pool("SRT", offset="(4,0,0)", to="(joint_sprase_coding-east)", height=10, depth=10, width=10, caption="Score Rank Threshold"),
    to_Sum("SRT_sum3", offset="(0,4,0)",to="(SRT-north)", radius=0, opacity=0.0),
    to_Pool("WT", offset="(4,0,0)", to="(SRT-east)", height=10, depth=10, width=10, caption="Workk Threshold"),
    to_Pool("AOR", offset="(4,0,0)", to="(WT-east)", height=10, depth=5, width=5, caption="Accept or Refuse"),


    # face_connection
    to_connection("face_feature_vector", "gallery"),
    to_connection("gallery", "face_sprase_coding"),
    to_connection("face_sprase_coding", "face_sum1"),
    to_connection("face_sum1", "face_sum2"),

    # ear_connection
    to_connection("ear_feature_vector", "ear_gallery"),
    to_connection("ear_gallery", "ear_sprase_coding"),
    to_connection("ear_sprase_coding", "ear_sum1"),
    to_connection("ear_sum1", "ear_sum2"),

    # fusion_connection
    to_connection("fusion_feature_vector", "fusion_gallery"),
    to_connection("fusion_gallery", "joint_sprase_coding"),
    to_connection("joint_sprase_coding", "SRT"),
    to_connection("SRT","WT"),

    to_connection("WT", "AOR"),
    to_skip("SRT", "AOR", pos=3),
     
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
