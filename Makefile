pre:
	pip install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7
	mkdir -p thirdparty
	git clone https://github.com/open-mmlab/mmdetection.git thirdparty/mmdetection
	cd thirdparty/mmdetection && git checkout v2.26.0
	pip install -U openmim
	mim install mmcv-full
	cd thirdparty/mmdetection && python -m pip install -e .
	python -m pip install -r requirements.txt

install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info
