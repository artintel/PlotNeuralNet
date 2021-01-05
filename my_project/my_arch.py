
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
    to_Conv("conv1", 1024, 64, offset="(0,0,0)", to="(0,0,0)", height=36, depth=36, width=2 ),
    *block_2ConvPool( name='b2', botton='conv1', top='pool_b2', s_filer=512, n_filer=128, offset="(1,0,0)", size=(30,30,3.5), opacity=0.5 ),
    *block_2ConvPool( name='b3', botton='pool_b2', top='pool_b3', s_filer=512, n_filer=128, offset="(1,0,0)", size=(30,30,3.5), opacity=0.5 ),
    *block_2ConvPool( name='b4', botton='pool_b3', top='pool_b4', s_filer=512, n_filer=128, offset="(1,0,0)", size=(30,30,3.5), opacity=0.5 ),
    to_Conv("conv2", s_filer=256, n_filer=1, offset="(3.5,0,0)", to="(pool_b4-east)", height=20, depth=20, width=1, caption="face-feature"  ),
    to_Sum("sum1", offset="(3.3,0,0)", to="(conv2-east)", radius=0, opacity=0.0),
    to_Sum("sum2", offset="(0,-7.0,0)",to="(sum1-south)", radius=0, opacity=0.0),
    to_Conv("conv3", 256, 1, offset="(4.5,0,0)", to="(sum1-east)", height=20, depth=20, width=1, caption="face-feature" ),
    *block_2ConvPool( name='b5', botton='conv3', top='pool_b5', s_filer=256, n_filer=1, offset="(2,0,0)", size=(20,20,1), opacity=0.5 ),
    *block_2ConvPool( name='b6', botton='pool_b5', top='pool_b6', s_filer=256, n_filer=1, offset="(2,0,0)", size=(20,20,1), opacity=0.5 ),
    to_SoftMax("Arcface_Loss", 200 ,"(3,0,0)", "(pool_b6-east)", caption="FC-layer"  ),
    to_Sum("sum4", offset="(3,0,0)", to="(Arcface_Loss-east)", radius=2, opacity=0.0, caption="..."),
    to_Gallery("gallery", offset="(3,0,0)", to="(sum4-east)", height=26, depth=26, width=13, caption="faces-gallery" ),
    to_Gallery_map("gallery_map_m1", offset="(-2.3,0,0)", to="(gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("gallery_map_m2", offset="(-1.5,0,0)", to="(gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("gallery_map_m3", offset="(-0.8,0,0)", to="(gallery-east)", height=24, depth=24, width=2.5 ),
    to_Conv("face_sprase_coding", 1, 200, offset="(3,0,0)", to="(gallery-east)", width=25, height=3, depth=1.5, caption="face-sprase-coding"),
    to_Sum("face_sum1", offset="(5,0,0)", to="(face_sprase_coding-east)", radius=0, opacity=0.0),
    to_Sum("face_sum2", offset="(0,-6.5,0)",to="(face_sum1-south)", radius=0, opacity=0.0),
    
    
    # ear-pipeline
    to_Conv("ear_conv1", 1024, 64, offset="(0,-15,0)", to="(0,0,0)", height=36, depth=36, width=2 ),
    *block_2ConvPool( name='ear_b2', botton='ear_conv1', top='ear_pool_b2', s_filer=512, n_filer=128, offset="(1,0,0)", size=(30,30,3.5), opacity=0.5 ),
    *block_2ConvPool( name='ear_b3', botton='ear_pool_b2', top='ear_pool_b3', s_filer=512, n_filer=128, offset="(1,0,0)", size=(30,30,3.5), opacity=0.5 ),
    *block_2ConvPool( name='ear_b4', botton='ear_pool_b3', top='ear_pool_b4', s_filer=512, n_filer=128, offset="(1,0,0)", size=(30,30,3.5), opacity=0.5 ),
    to_Conv("ear_conv2", s_filer=256, n_filer=1, offset="(3.5,0,0)", to="(ear_pool_b4-east)", height=20, depth=20, width=1, caption="ear-feature"  ),
    to_Sum("ear_sum1", offset="(3.5,0,0)", to="(ear_conv2-west)", radius=0, opacity=0.0),
    to_Sum("ear_sum2", offset="(0,6.45,0)",to="(ear_sum1-north)", radius=0, opacity=0.0),
    to_Conv("ear_conv3", 256, 1, offset="(4.5,0,0)", to="(ear_sum1-east)", height=20, depth=20, width=1, caption="ear-feature" ),
    *block_2ConvPool( name='ear_b5', botton='ear_conv3', top='ear_pool_b5', s_filer=256, n_filer=1, offset="(2,0,0)", size=(20,20,1), opacity=0.5 ),
    *block_2ConvPool( name='ear_b6', botton='ear_pool_b5', top='ear_pool_b6', s_filer=256, n_filer=1, offset="(2,0,0)", size=(20,20,1), opacity=0.5 ),
    to_SoftMax("ear_Arcface_Loss", 200 ,"(3,0,0)", "(ear_pool_b6-east)", caption="FC-layer" ),
    to_Sum("ear_sum4", offset="(3,0,0)", to="(ear_Arcface_Loss-east)", radius=2, opacity=0.0, caption="..."),
    to_Gallery("ear_gallery", offset="(3,0,0)", to="(ear_sum4-east)", height=26, depth=26, width=13, caption="faces-gallery" ),
    to_Gallery_map("ear_gallery_map_m1", offset="(-2.3,0,0)", to="(ear_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("ear_gallery_map_m2", offset="(-1.5,0,0)", to="(ear_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("ear_gallery_map_m3", offset="(-0.8,0,0)", to="(ear_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Conv("ear_sprase_coding", 1, 200, offset="(3,0,0)", to="(ear_gallery-east)", width=25, height=3, depth=1.5, caption="ear-sprase-coding"),
    to_Sum("ear_sum5", offset="(5,0,0)", to="(ear_sprase_coding-east)", radius=0, opacity=0.0),
    to_Sum("ear_sum6", offset="(0,5.85,0)",to="(ear_sum5-south)", radius=0, opacity=0.0),

    # fusion-pipeline
    to_Sum("sum3", offset="(0,-0.78,0)", to="(sum2-south)", radius=4, opacity=0.6 ),
    to_Fusion_Conv("fusion_conv3", s_filer=256, n_filer=1, offset="(3.5,0,0)", to="(sum3-east)", height=20, depth=20, width=1, caption="fusion-feature" ),
    *fusion_block_2ConvPool( name='fusion_b1', botton='fusion_conv3', top='fusion_pool_b1', s_filer=256, n_filer=1, offset="(2,0,0)", size=(20,20,1), opacity=0.5 ),
    *fusion_block_2ConvPool( name='fusion_b2', botton='fusion_pool_b1', top='fusion_pool_b2', s_filer=256, n_filer=1, offset="(2,0,0)", size=(20,20,1), opacity=0.5 ),
    to_SoftMax("fusion_Arcface_Loss", 200 ,"(3,0,0)", "(fusion_pool_b2-east)", caption="FC-layer"),
    to_Sum("fusion_sum4", offset="(3,0,0)", to="(fusion_Arcface_Loss-east)", radius=2, opacity=0.0, caption="..."),
    to_Gallery("fusion_gallery", offset="(3,0,0)", to="(fusion_sum4-east)", height=26, depth=26, width=13, caption="fusion-gallery" ),
    to_Gallery_map("fusion_gallery_map_m1", offset="(-2.3,0,0)", to="(fusion_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("fusion_gallery_map_m2", offset="(-1.5,0,0)", to="(fusion_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Gallery_map("fusion_gallery_map_m3", offset="(-0.8,0,0)", to="(fusion_gallery-east)", height=24, depth=24, width=2.5 ),
    to_Conv("joint_sprase_coding", 1, 200, offset="(3,0,0)", to="(fusion_gallery-east)", width=25, height=3, depth=1.5, caption="joint-sprase-coding"),
    to_Pool("SRT", offset="(4,0,0)", to="(joint_sprase_coding-east)", height=10, depth=10, width=10, caption="Score Rank Threshold"),

    to_Pool("WT", offset="(4,0,0)", to="(SRT-east)", height=10, depth=10, width=10, caption="Workk Threshold"),
    to_Pool("AOR", offset="(4,0,0)", to="(WT-east)", height=10, depth=5, width=5, caption="Accept or Refuse"),

    # face_connection
    to_connection("pool_b2", "conv2"),
    to_connection("conv2", "sum1"),
    to_connection("sum1", "sum2"),
    to_connection("sum1", "conv3"),
    to_connection("pool_b6", "Arcface_Loss"),
    to_connection("Arcface_Loss", "sum4"),
    to_connection("sum4", "gallery"),
    to_connection("gallery", "face_sprase_coding"),
    to_connection("face_sprase_coding", "face_sum1"),
    to_connection("face_sum1", "face_sum2"),

    # ear_connection
    to_connection("ear_pool_b2", "ear_conv2"),
    to_connection("ear_conv2", "ear_sum1"),
    to_connection("ear_sum1", "ear_sum2"),
    to_connection("ear_sum1", "ear_conv3"),
    to_connection("ear_pool_b6", "ear_Arcface_Loss"),
    to_connection("ear_Arcface_Loss", "ear_sum4"),
    to_connection("ear_sum4", "ear_gallery"),
    to_connection("ear_gallery", "ear_sprase_coding"),
    to_connection("ear_sprase_coding", "ear_sum5"),
    to_connection("ear_sum5", "ear_sum6"),

    # fusion_connection
    to_connection("sum3", "fusion_conv3"),
    to_connection("fusion_pool_b1", "fusion_Arcface_Loss"),
    to_connection("fusion_Arcface_Loss", "fusion_sum4"),
    to_connection("fusion_sum4", "fusion_gallery"),
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
    
