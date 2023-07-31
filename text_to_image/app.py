import gradio as gr
import os
import random
import time
from zipfile import ZipFile
import tempfile
import string
import time



import argparse
import ast
import gradio as gr
from os.path import isdir
#from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils
from utils import dataset_utils
from utils import gradio_utils as gr_utils
import widgets
import app as ap
from app import load_or_prepare_widgets


logs = utils.prepare_logging(__file__)

# Utility for sidebar description and selection of the dataset
DATASET_NAME_TO_DICT = dataset_utils.get_dataset_info_dicts()

directory = tempfile.mkdtemp(dir="./")

imagens = []

model = gr.Interface.load(
	"models/dreamlike-art/dreamlike-photoreal-2.0",
)

#o = os.getenv("P")
o = "V"

m_out = ("""
<div id="restart">
<h3 id="head">Loading Time Limit Reached.</h3><br>
<h4 id="cont">Please choose a Simpler Prompt, or <b>Upgrade</b> for faster loading.</h4>
</div>
""")
loading=("""
<div class="lds-ellipsis"><div></div><div></div><div></div><div></div></div>""")

def add_random_noise(prompt, noise_level=1.00):
	if noise_level == 0:
		noise_level = 0.00
	if noise_level == None:
		noise_level = 1.00
	percentage_noise = noise_level * 5
	num_noise_chars = int(len(prompt) * (percentage_noise/100))
	noise_indices = random.sample(range(len(prompt)), num_noise_chars)
	prompt_list = list(prompt)
	noise_chars = list(string.ascii_letters + string.punctuation + ' ' + string.digits)
	noise_chars.extend(['😍', '💩', '😂', '🤔', '😊', '🤗', '😭', '🙄', '😷', '🤯', '🤫', '🥴', '😴', '🤩', '🥳', '😔', '😩', '🤪', '😇', '🤢', '😈', '👹', '👻', '🤖', '👽', '💀', '🎃', '🎅', '🎄', '🎁', '🎂', '🎉', '🎈', '🎊', '🎮', '❤️', '💔', '💕', '💖', '💗', '🐶', '🐱', '🐭', '🐹', '🦊', '🐻', '🐨', '🐯', '🦁', '🐘', '🔥', '🌧️', '🌞', '🌈', '💥', '🌴', '🌊', '🌺', '🌻', '🌸', '🎨', '🌅', '🌌', '☁️', '⛈️', '❄️', '☀️', '🌤️', '⛅️', '🌥️', '🌦️', '🌧️', '🌩️', '🌨️', '🌫️', '☔️', '🌬️', '💨', '🌪️', '🌈'])
	for index in noise_indices:
		prompt_list[index] = random.choice(noise_chars)
	return "".join(prompt_list)

def build():
	def zip_files():
		zip_name = f"{b.prompt.split(' ')[0]}_{random.randint(0, 10000)}.zip"
		with ZipFile(zip_name, "w") as zipObj:
			for file in b.imagens:
				zipObj.write(file, os.path.basename(file))
		b.imagens = []
		return zip_name
	def clear():
		return gr.update(value=0),gr.update(value=0)
	def start():
		stamp = time.time()
		return gr.update(value=stamp),gr.update(value=0)
	def end(stamp):
		ts = stamp + 360
		ti = time.time()
		if ti > ts and stamp != 0:
			return gr.update(value=1),gr.HTML.update(f"{m_out}",visible=True)
		else:
			return gr.update(value=0),None
	def im_fn(prompt,noise_level,h=None):
		try:
			if h == o:
				prompt_with_noise = add_random_noise(prompt, noise_level)
				imagem = model(prompt_with_noise)
				b.prompt = prompt
				b.imagens.append(imagem)
				return imagem
			elif h != o:
				return(None,None)
		except Exception as E:
			return None, None 
	def cl_fac():
		return "",gr.HTML.update(f"{loading}")
	with gr.Blocks() as b:
		b.imagens: list = []
		with gr.Row():
			with gr.Column():
				prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt")
				noise_level = gr.Slider(minimum=0.0, maximum=10, step=0.1, label="Noise Level between images.")
			with gr.Column():
				with gr.Row():
					btn1 = gr.Button("Generate")
					btn2 = gr.Button("Clear")
		message=gr.HTML("<div></div>")
		message2=gr.HTML("",visible=False)

		with gr.Row():
			out1 = gr.Image()
			out2 = gr.Image()
		with gr.Row():
			out3 = gr.Image()
			out4 = gr.Image()
		with gr.Row():
			out5 = gr.Image()
			out6 = gr.Image()
		with gr.Row():
			# btn3 = gr.Button("Download")
			caixa = gr.File(file_count="multiple", file_types=["text", ".json", ".csv", "image"])

		with gr.Row(visible=False):
			h_variavel=gr.Textbox(value="V")
			t_state=gr.Number()
			t_switch=gr.Textbox(value=0)
			auto= gr.Image()
		def clear_all():
			return "",None,None,None,None,None,None,None,None,1,gr.HTML.update("<div></div>")
		fac_b = gr.Textbox(value="",visible=False)

		def noth():
			return gr.HTML.update("<div></div>")
		#a1=btn1.click(noth,None,btn1,every=1)
		btn1.click(cl_fac,None,[fac_b,message],show_progress=False)
		b1=btn1.click(start,None,[t_state,t_switch],show_progress=True)
		sta = t_state.change(end,t_state,[t_switch,message2],every=1,show_progress=True)
		b2=btn1.click(im_fn,[prompt,noise_level,h_variavel],[out1,], show_progress=True)
		b3=out1.change(im_fn,[prompt,noise_level,h_variavel],[out2,], show_progress=True)
		b4=out2.change(im_fn,[prompt,noise_level,h_variavel],[out3,], show_progress=True)
		b5=out3.change(im_fn,[prompt,noise_level,h_variavel],[out4,], show_progress=True)
		b6=out4.change(im_fn,[prompt,noise_level,h_variavel],[out5,], show_progress=True)
		b7=out5.change(im_fn,[prompt,noise_level,h_variavel],[out6], show_progress=True)
		b8=out6.change(noth,None,[message], show_progress=False)
		b8=out6.change(zip_files,None,[caixa], show_progress=False)
		swi=t_switch.change(clear,None,[t_switch,fac_b], cancels=[sta,b2,b3,b4,b5,b6,b7],show_progress=False)
		#btn2.click(noth,None,message,cancels=[b1,sta,b2,b3,b4,b5,swi],show_progress=False)
		btn2.click(clear_all, None,[fac_b,prompt,out1,out2,out3,out4,out5,out6,t_state,t_switch,message],cancels=[b1,sta,b2,b3,b4,b5,b6,b7,b8,swi],show_progress=False)
		# btn3.click(zip_files,None,[caixa],show_progress=False)
		# caixa.change(noth,None,[message],show_progress=False)
	b.queue(concurrency_count=100).launch(show_api=False)
build()


## check that it works with a text prompt as variables parsed into build()