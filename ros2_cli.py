import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

# ros things
import rclpy
from rclpy.node import Node
# from straps_msgs.srv import MGM
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge
from PIL import Image


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class llava_node(Node):
    def __init__(self, args):
        super().__init__('{}/llava_node'.format(args.ros_namespace))

        # ROS parameters
        # self.declare_parameter('robot_prefix', 'mt001')  # Default value for robot_prefix
        # self.declare_parameter('sensor_index', '1')
        # self.declare_parameter('debugging', True)

        # self.robot_prefix = self.get_parameter('robot_prefix').get_parameter_value().string_value
        # self.sensor_index = self.get_parameter('sensor_index').get_parameter_value().string_value
    

        self.model_name = get_model_name_from_path(args.model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, args.model_base, self.model_name, args.load_8bit, args.load_4bit, device=args.device)
        
        self.conv_mode = None

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if args.conv_mode is not None and self.conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(self.conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = self.conv_mode

        
        self.roles = {}

        
        self.args = args

        self.current_compressed_img = CompressedImage()
        self.bridge = CvBridge()
        # self.mgm_srv = self.create_service(MGM, '/mgm_interface', self.mgm_callback)
        self.questions = []
        self.questions.append("Is his eyes opened?")
        self.questions.append("Is this a real human?")

        self.idx = 0
        self.image = load_image(args.image_file)

        timer_period = 1.0 #1hz        
        self.timer = self.create_timer(timer_period, self.timer_callback)
        print("init !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def timer_callback(self):

        print("in call back")
 
        image_size = self.image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([self.image], self.image_processor, self.model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to( self.model.device, dtype=torch.float16)

        for inp in self.questions:

            if self.image is not None:
                # first message
                if self.model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                self.image = None
            
            self.conv.append_message( self.conv.roles[0], inp )
            self.conv.append_message( self.conv.roles[1], None)
            prompt = self.conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # streamer = None
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)

            outputs = self.tokenizer.decode(output_ids[0]).strip()
            self.conv.messages[-1][-1] = outputs

            # if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


def main(args= None):
    rclpy.init(args=None)

    node = llava_node(args)
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--ros_namespace", type=str, default="mt001" )
    parser.add_argument("--sensor_idx", type=int, default=1 )    
    args = parser.parse_args()
    main(args)
