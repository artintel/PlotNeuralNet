
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
    to_Conv("conv1", s_filer=256, n_filer=1, offset="(0,0,0)", to="(0,0,0)", height=15, depth=15, width=6, caption="face-feature-map"  ),
    to_Sum("sum1", offset="(3.3,0,0)", to="(conv1-east)", radius=0, opacity=0.0),
    to_Sum("sum2", offset="(0,-6.1,0)",to="(sum1-south)", radius=0, opacity=0.0),
    to_Conv("conv3", 256, 1, offset="(4.5,0,0)", to="(sum1-east)", height=15, depth=15, width=6, caption="face-feature-map" ),

    
    
    # # ear-pipeline
    to_Conv("ear_conv1", s_filer=256, n_filer=1, offset="(0,-13,0)", to="(0,0,0)", height=15, depth=15, width=6, caption="ear-feature-map"  ),
    to_Sum("ear_sum1", offset="(4.5,0,0)", to="(ear_conv1-west)", radius=0, opacity=0.0),
    to_Sum("ear_sum2", offset="(0,6.1,0)",to="(ear_sum1-north)", radius=0, opacity=0.0),
    to_Conv("ear_conv3", 256, 1, offset="(4.5,0,0)", to="(ear_sum1-east)", height=15, depth=15, width=6, caption="ear-feature-map" ),

    # # fusion-pipeline
    to_Sum("sum3", offset="(0,-6.5,0)", to="(sum1-south)", radius=2, opacity=0.6 ),
    to_Fusion_Conv("fusion_conv3", s_filer=256, n_filer=1, offset="(4.0,0,0)", to="(sum3-east)", height=15, depth=15, width=6, caption="fusion-feature-map" ),

    # # # face_connection
    to_connection("conv1", "sum1"),
    to_connection("sum1", "sum2"),
    to_connection("sum1", "conv3"),

    # # # ear_connection
    to_connection("ear_conv1", "ear_sum1"),
    to_connection("ear_sum1", "ear_sum2"),
    to_connection("ear_sum1", "ear_conv3"),

    # # # fusion_connection
    to_connection("sum3", "fusion_conv3"),
     
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()