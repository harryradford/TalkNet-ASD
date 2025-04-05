import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features, multiprocessing, copy

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--inputVideo',            type=str, default=None,   help='Path to input video file (if not in demo folder)')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
parser.add_argument('--ffmpegPath',            type=str, default="ffmpeg", help='Path to FFmpeg binary')
parser.add_argument('--startTime',             type=float, default=None, help='Start time in seconds for processing a portion of the video')
parser.add_argument('--endTime',               type=float, default=None, help='End time in seconds for processing a portion of the video')
parser.add_argument('--outputCoords',          type=str, default=None,   help='Path to output JSON file containing active speaker coordinates')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--frameSkip',             type=int,   default=1,    help='Process every Nth frame for face detection')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/data08/col",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

if args.evalCol == True:
	# The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
	# 	              2. extract audio, extract video frames
	#                 3. scend detection, face detection and face tracking
	#                 4. active speaker detection for the detected face clips
	#                 5. use iou to find the identity of each face clips, compute the F1 results
	# The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
	# The step 4 and 5 need less than 10 minutes
	# Need about 20G space finally
	# ```
	args.videoName = 'col'
	args.videoFolder = args.colSavePath
	args.savePath = os.path.join(args.videoFolder, args.videoName)
	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.duration = 0
	if os.path.isfile(args.videoPath) == False:  # Download video
		link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
		cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
		output = subprocess.call(cmd, shell=True, stdout=None)
	if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
		link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
		cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
		subprocess.call(cmd, shell=True, stdout=None)
		cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
		subprocess.call(cmd, shell=True, stdout=None)
		os.remove(args.videoFolder + '/col_labels.tar.gz')	
else:
	if args.inputVideo is not None:
		# Use the provided input video path
		args.videoPath = args.inputVideo
		# Set savePath to a directory with the same name as the video (without extension)
		video_name = os.path.splitext(os.path.basename(args.inputVideo))[0]
		args.savePath = os.path.join(args.videoFolder, video_name)
	else:
		# Use the existing behavior as fallback
		args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
		args.savePath = os.path.join(args.videoFolder, args.videoName)

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cpu')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		# Skip frames based on frameSkip parameter
		if fidx % args.frameSkip != 0:
			dets.append([])  # Add empty detection for skipped frames
			continue
			
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("%s -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.ffmpegPath, args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("%s -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (args.ffmpegPath, cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cpu()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cpu()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    # Get all frames from all segments
    all_frames = []
    for segment_dir in sorted(glob.glob(os.path.join(args.savePath, 'segment_*'))):
        frames = glob.glob(os.path.join(segment_dir, 'pyframes', '*.jpg'))
        frames.sort()
        all_frames.extend(frames)
    
    faces = [[] for i in range(len(all_frames))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            if frame < len(faces):  # Ensure frame index is valid
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = numpy.mean(s)
                faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
    
    # Save coordinates to JSON if output path is specified
    if args.outputCoords is not None:
        import json
        # Convert numpy types to Python native types for JSON serialization
        coords_data = []
        for frame_idx, frame_faces in enumerate(faces):
            frame_data = {
                'frame': frame_idx,
                'faces': []
            }
            for face in frame_faces:
                face_data = {
                    'track_id': int(face['track']),
                    'score': float(face['score']),
                    'x': float(face['x']),
                    'y': float(face['y']),
                    'size': float(face['s'])
                }
                frame_data['faces'].append(face_data)
            coords_data.append(frame_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.outputCoords), exist_ok=True)
        
        # Save to JSON file
        with open(args.outputCoords, 'w') as f:
            json.dump(coords_data, f, indent=2)
    
    firstImage = cv2.imread(all_frames[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
    colorDict = {0: 0, 1: 255}
    
    for fidx, fname in tqdm.tqdm(enumerate(all_frames), total = len(all_frames)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face['score'] >= 0))]
            txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
            cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        vOut.write(image)
    vOut.release()
    
    # Use the original audio from the input video, but only for the specified time range
    start_time = args.startTime if args.startTime is not None else 0
    end_time = args.endTime if args.endTime is not None else None
    
    # Create a temporary audio file for the specified time range
    temp_audio = os.path.join(args.pyaviPath, 'temp_audio.wav')
    if end_time is not None:
        command = ("%s -y -i %s -threads %d -ss %.3f -to %.3f -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel panic" % \
            (args.ffmpegPath, args.videoPath, args.nDataLoaderThread, start_time, end_time, temp_audio))
    else:
        command = ("%s -y -i %s -threads %d -ss %.3f -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel panic" % \
            (args.ffmpegPath, args.videoPath, args.nDataLoaderThread, start_time, temp_audio))
    subprocess.call(command, shell=True, stdout=None)
    
    # Combine video with the cropped audio
    command = ("%s -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (args.ffmpegPath, os.path.join(args.pyaviPath, 'video_only.avi'), temp_audio, \
        args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
    output = subprocess.call(command, shell=True, stdout=None)
    
    # Clean up temporary audio file
    if os.path.exists(temp_audio):
        os.remove(temp_audio)

def evaluate_col_ASD(tracks, scores, args):
	txtPath = args.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])
			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))	  

def split_video_into_segments(args):
    """Split video into segments based on CPU cores"""
    # Get video duration
    cap = cv2.VideoCapture(args.videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    cap.release()
    
    # Handle start and end times
    start_time = args.startTime if args.startTime is not None else 0
    end_time = args.endTime if args.endTime is not None else total_duration
    
    # Validate time range
    if start_time < 0:
        start_time = 0
    if end_time > total_duration:
        end_time = total_duration
    if start_time >= end_time:
        raise ValueError("Start time must be less than end time")
    
    # Calculate duration of the portion to process
    duration = end_time - start_time
    
    # Calculate number of segments based on CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # For 1-minute videos, limit to 6 threads maximum
    # This provides a good balance between speed and overhead
    # Leaves more CPU resources for other applications
    max_threads = 6
    num_segments = min(max(1, num_cores - 1), max_threads)
    
    # Calculate segment duration
    segment_duration = duration / num_segments
    
    # Create segment arguments
    segments = []
    for i in range(num_segments):
        segment_start = start_time + (i * segment_duration)
        segment_end = start_time + ((i + 1) * segment_duration) if i < num_segments - 1 else end_time
        
        # Create a copy of args for this segment
        segment_args = copy.deepcopy(args)
        segment_args.start = segment_start
        segment_args.duration = segment_end - segment_start
        segment_args.savePath = os.path.join(args.savePath, f'segment_{i}')
        segment_args.pyaviPath = os.path.join(segment_args.savePath, 'pyavi')
        segment_args.pyframesPath = os.path.join(segment_args.savePath, 'pyframes')
        segment_args.pyworkPath = os.path.join(segment_args.savePath, 'pywork')
        segment_args.pycropPath = os.path.join(segment_args.savePath, 'pycrop')
        segment_args.videoFilePath = os.path.join(segment_args.pyaviPath, 'video.avi')
        segment_args.audioFilePath = os.path.join(segment_args.pyaviPath, 'audio.wav')
        segments.append(segment_args)
    
    return segments

def process_segment(segment_args):
    """Process a single video segment"""
    # Create necessary directories
    os.makedirs(segment_args.pyaviPath, exist_ok=True)
    os.makedirs(segment_args.pyframesPath, exist_ok=True)
    os.makedirs(segment_args.pyworkPath, exist_ok=True)
    os.makedirs(segment_args.pycropPath, exist_ok=True)
    
    # Extract frames for this segment
    command = ("%s -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
        (segment_args.ffmpegPath, segment_args.videoPath, segment_args.nDataLoaderThread, segment_args.start, 
         segment_args.start + segment_args.duration, segment_args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    
    # Extract audio
    command = ("%s -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (segment_args.ffmpegPath, segment_args.videoFilePath, segment_args.nDataLoaderThread, segment_args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    
    # Extract frames
    command = ("%s -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
        (segment_args.ffmpegPath, segment_args.videoFilePath, segment_args.nDataLoaderThread, 
         os.path.join(segment_args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    
    # Process the segment
    scene = scene_detect(segment_args)
    faces = inference_video(segment_args)
    
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= segment_args.minTrack:
            allTracks.extend(track_shot(segment_args, faces[shot[0].frame_num:shot[1].frame_num]))
    
    vidTracks = []
    for ii, track in enumerate(allTracks):
        vidTracks.append(crop_video(segment_args, track, os.path.join(segment_args.pycropPath, '%05d'%ii)))
    
    # Evaluate network for this segment
    files = glob.glob("%s/*.avi"%segment_args.pycropPath)
    files.sort()
    scores = evaluate_network(files, segment_args)
    
    return {
        'tracks': vidTracks,
        'scores': scores,
        'start_time': segment_args.start,
        'duration': segment_args.duration,
        'segment_args': segment_args  # Include the segment arguments
    }

def merge_segments(segment_results):
    """Merge results from parallel segments"""
    all_tracks = []
    all_scores = []
    
    # Sort segments by start time
    segment_results.sort(key=lambda x: x['start_time'])
    
    # Calculate frame offset for each segment
    frame_offset = 0
    for result in segment_results:
        # Adjust frame indices in tracks
        for track in result['tracks']:
            # Create a copy of the track to avoid modifying the original
            adjusted_track = copy.deepcopy(track)
            # Adjust frame indices
            adjusted_track['track']['frame'] = track['track']['frame'] + frame_offset
            all_tracks.append(adjusted_track)
        
        # Add scores
        all_scores.extend(result['scores'])
        
        # Update frame offset for next segment
        frames_in_segment = len(glob.glob(os.path.join(result['segment_args'].pyframesPath, '*.jpg')))
        frame_offset += frames_in_segment
    
    return all_tracks, all_scores

def main():
    # Initialization 
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    os.makedirs(args.pyaviPath, exist_ok=True)
    os.makedirs(args.pyframesPath, exist_ok=True)
    os.makedirs(args.pyworkPath, exist_ok=True)
    os.makedirs(args.pycropPath, exist_ok=True)
    
    # Split video into segments
    segments = split_video_into_segments(args)
    
    # Process segments in parallel
    with multiprocessing.Pool(len(segments)) as pool:
        segment_results = list(tqdm.tqdm(
            pool.imap(process_segment, segments),
            total=len(segments),
            desc="Processing segments"
        ))
    
    # Merge results
    all_tracks, all_scores = merge_segments(segment_results)
    
    # Save final results
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(all_tracks, fil)
    
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(all_scores, fil)
    
    # Visualize results
    if args.evalCol == True:
        evaluate_col_ASD(all_tracks, all_scores, args)
    else:
        visualization(all_tracks, all_scores, args)

if __name__ == '__main__':
    main()
