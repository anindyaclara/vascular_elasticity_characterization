import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import natsort
from skimage.measure import label as sklabel, regionprops
import sys
import shutil
import csv, os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Grayscale
import torchvision.transforms as transforms
import glob
import re
import base64
import subprocess
import time
import random
from matplotlib.ticker import MultipleLocator, FuncFormatter

#DEFINE RESNET-UNET
#Double Conv
class Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Added padding=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Added padding=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)

#Resnet Encoder
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Gunakan weight enum agar tidak ada warning
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)
        
        # ResNet34 feature dimensions: 64, 64, 128, 256, 512
        # We need to adapt these to match your UNet: 64, 128, 256, 512, 1024
        
        # Initial layers
        self.first = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # Output: 64 channels
        
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # Output: 64 channels
        self.layer2 = resnet.layer2  # Output: 128 channels
        self.layer3 = resnet.layer3  # Output: 256 channels
        self.layer4 = resnet.layer4  # Output: 512 channels
        
        # Additional layer to get to 1024 channels
        self.extra = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Store intermediate outputs for skip connections
        features = []
        
        # Initial block - 64 channels
        x0 = self.first(x)
        
        # Add maxpool output
        x = self.maxpool(x0)
        
        # Block 1 - 64 channels
        x1 = self.layer1(x)
        features.append(x1)  # First feature: 64 channels
        
        # Block 2 - 128 channels
        x2 = self.layer2(x1)
        features.append(x2)  # Second feature: 128 channels
        
        # Block 3 - 256 channels
        x3 = self.layer3(x2)
        features.append(x3)  # Third feature: 256 channels
        
        # Block 4 - 512 channels
        x4 = self.layer4(x3)
        features.append(x4)  # Fourth feature: 512 channels
        
        # Extra block - 1024 channels
        x5 = self.extra(x4)
        features.append(x5)  # Fifth feature: 1024 channels
        
        return features

#Define Decoder
import torchvision
class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64)): #hpus 1024
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

#Gabungkan Encoder-Decoder
class ResNetUNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), #hpus1024
                 dec_chs=(1024, 512, 256, 128, 64), 
                 num_class=2, pretrained=True, 
                 retain_dim=False, out_sz=(192, 192)):
        super().__init__()
        
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # Use your original decoder
        self.decoder = Decoder(dec_chs)
        
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz
        
        # For handling grayscale inputs
        self.input_layer = nn.Conv2d(1, 3, kernel_size=1)
        
    def forward(self, x):
        # Convert grayscale to 3 channel if needed
        if x.shape[1] == 1:
            x = self.input_layer(x)
            
        # Get features from encoder
        enc_ftrs = self.encoder(x)
        
        # Pass to decoder (making sure dimensions match your original implementation)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        
        # Final classification layer
        out = self.head(out)
        
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz, mode='bilinear', align_corners=True)
            
        return out
#Buat ngesort
def natural_sort_key(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers] if numbers else [0]
# Inisialisasi session state
if "animated" not in st.session_state:
    st.session_state.animated = False

def main():
    # Tampilkan logo di atas sidebar
    with st.sidebar:
        # Baca dan encode kedua logo
        logo1 = base64.b64encode(open("./our_team/logoits.png", "rb").read()).decode()
        logo2 = base64.b64encode(open("./our_team/logobme.png", "rb").read()).decode()

        # HTML untuk menampilkan dua logo sejajar
        st.markdown(
            f"""
            <div style='display: flex; justify-content: center; align-items: center; gap: 10px;'>
                <img src='data:image/png;base64,{logo1}' width='120'>
                <img src='data:image/png;base64,{logo2}' width='80'>
            </div>
            <hr style='margin-top: 10px; margin-bottom: 30px;'>
            """,
            unsafe_allow_html=True
        )
    menu = ["About", "Our Team", "Elasticity Characterization", "Image Processing"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.markdown("<h1 style='text-align: center;'>PULSE-TEC</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Portable Ultrasound with Sensor-based Elasticity TEChnology</h4>", unsafe_allow_html=True)
        # Muat gambar dari file lokal (ganti dengan path gambarmu)
        image = Image.open("gambarst.png")
        st.image(image, caption="PULSE-TEC Device", use_column_width=True)
        st.markdown("""
        <div style='text-align: justify; font-size: 16px'>
            <span style='font-weight: bold; font-size: 30px;'>This innovative system</span> combines safe imaging capability of a portable ultrasound with a loadcell-based pressure sensor to evaluate <b>vascular elasticity</b> — a critical biomarker in assessing cardiovascular health.  
            By integrating controlled pressure application and diameter tracking, PULSE-TEC enables quantifiable elasticity measurements, offering valuable insights into arterial stiffness and potential early signs of vascular disease.
            <br><br>
            The device is designed to be <b>compact, accessible, and highly informative</b> for both research and clinical applications.
        </div>
        """, unsafe_allow_html=True)
        
    elif choice == "Our Team":
        # === Baris atas: dua gambar sejajar dengan teks di tengah ===
        col1, spacer, col2 = st.columns([1, 0.5, 1])

        with col1:
            img1 = base64.b64encode(open("./our_team/team1.jpg", "rb").read()).decode()
            st.markdown(
                f"""
                <div style='width: 300px; margin: auto; text-align: center;'>
                    <img src='data:image/png;base64,{img1}' width='300'>
                    <div style='margin-top: 10px;'>
                        <div style='font-size: 17px; font-weight: bold;'>Prof. Dr. Tri Arief Sardjono, S.T., M.T.</div>
                        <div style='font-size: 16px;'>Supervisor</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            img2 = base64.b64encode(open("./our_team/team2.jpg", "rb").read()).decode()
            st.markdown(
                f"""
                <div style='width: 300px; margin: auto; text-align: center;'>
                    <img src='data:image/png;base64,{img2}' width='300'>
                    <div style='margin-top: 10px;'>
                        <div style='font-size: 17px; font-weight: bold;'>Dr. Norma Hermawan, S.T., M.T.</div>
                        <div style='font-size: 16px;'>Supervisor</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Spacer vertikal antara baris
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Gambar tengah bawah
        img3 = base64.b64encode(open("./our_team/team3.jpg", "rb").read()).decode()
        st.markdown(
            f"""
            <div style='display: flex; justify-content: center;'>
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{img3}' width='300'>
                    <p style='margin-top: 10px; font-size: 16px;'>
                        <strong style='font-size: 17px; font-weight: bold;'>Anindya Clarasanty</strong><br>Biomedical Engineering Student
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # === Deskripsi di bawah semua anggota tim ===
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: justify; font-size: 16px; max-width: 800px; margin: auto;'>
                <p>
                    <span style='font-weight: bold; font-size: 30px;'>Our team</span> is dedicated to advancing research in ultrasound imaging recognized as one of the safest and most accessible diagnostic modalities. 
                    We focus on exploring novel methods for analyzing vascular health, particularly through quantitative assessment of arterial elasticity and morphology. 
                    By integrating expertise in medical imaging, image processing, and software development, we aim to build intelligent, evidence-based tools that enhance clinical decision-making 
                    and support the early detection of cardiovascular conditions. Our interdisciplinary approach reflects a shared commitment to improving patient outcomes through innovation and research excellence.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif choice == "Elasticity Characterization":
        st.subheader("Elasticity Characterization")
        # Inisialisasi Pilihan Subjek
        if "choice" not in st.session_state:
           st.session_state.choice = "Subject1"
        Subject = []
        for i in range(1,50): #list subjek
            Subject.append('Subject'+str(i))
    
        choice_video = st.selectbox('Pick one sample video', Subject)
        st.session_state.choice = choice_video
        st.write('Here video of', choice_video)
        path_video = "./video_usg/%s.mp4" % st.session_state.choice
        try :
          st.video(path_video)
          vidcap1 = cv2.VideoCapture(path_video)
          duration1 = vidcap1.get(cv2.CAP_PROP_POS_MSEC)
        except:
          e = sys.exc_info()[0]
          print(st.write(e))
        
        #Pemrosesan Segmentasi
        start_processing = st.button('Start Processing')
        #Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "./resnet_iou_0.9545.pt"  

        model = ResNetUNet(num_class=2, pretrained=False, retain_dim=True, out_sz=(192, 192))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        #Load folder hasil
        os.makedirs(f'./segmen_images/{choice_video}', exist_ok=True)
        os.makedirs(f'./artery_height_images/{choice_video}', exist_ok=True)

        #Custom Data
        image_size = (384, 384)
        data_transform = Compose([
            Resize(image_size),
            Grayscale(),
            ToTensor(), #norm dr 0-255 ke 0-1
            Normalize(mean=[0.5], std=[0.5])  
        ])

        if start_processing:
            st.success('Image Processing Started')
            expander = st.expander("See process status")
            progress_segment = st.progress(0) #status proses

            raw_image_folder = f"./raw_images/{choice_video}"
            segmen_folder = f"./segmen_images/{choice_video}"
            radius_folder = f"./artery_height_images/{choice_video}"
            csv_folder = f"./arheight_measurements/{choice_video}"
            os.makedirs(segmen_folder, exist_ok=True) #pastikan ada
            os.makedirs(radius_folder, exist_ok=True)
            os.makedirs(csv_folder, exist_ok=True)

            image_paths = sorted(glob.glob(os.path.join(raw_image_folder, "*.png")))
            total_images = len(image_paths)
            frame_count = total_images
            st.write('frame_count = ', frame_count)

            arr_width_artery = []
            arr_height_artery = []
            arr_width_px = []
            arr_height_px = []

            csv_folder = f"./arheight_measurements/{choice_video}"
            os.makedirs(csv_folder, exist_ok=True)

            csv_file = os.path.join(csv_folder, "artery_geometry.csv")
            if not os.path.exists(csv_file):
            # Tulis header sekali di awal
                with open(csv_file, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Filename", "Width (mm)", "Height (mm)", "Width (px)", "Height (px)", "x1", "y1", "x2", "y2"])

            for count, image_path in enumerate(image_paths, start=1):
                original_filename = os.path.basename(image_path)
                segmen_output_path = os.path.join(segmen_folder, original_filename)
                radius_output_path = os.path.join(radius_folder, original_filename)

                # Skip frame kalau sudah diproses
                if os.path.exists(segmen_output_path) and os.path.exists(radius_output_path):
                    expander.info(f"Frame {count} sudah diproses, dilewati.")
                    progress_segment.progress(count / total_images)
                    continue

                expander.success(f"Processing frame {count}")
                progress_bar = expander.progress(0)

                image = cv2.imread(image_path)
                raw_image = image.copy()
                progress_bar.progress(10)

                # Baca dan convert ke grayscale
                image_rgb = Image.open(image_path).convert("RGB")
                gray_pil = image_rgb.convert("L")
                progress_bar.progress(20)

                # Transform
                processed_image = data_transform(gray_pil).unsqueeze(0).to(device)
                original_h, original_w = np.array(gray_pil).shape
                progress_bar.progress(40)

                # Segmentasi
                with torch.no_grad():
                    output = model(processed_image)
                    probs = torch.softmax(output, dim=1)
                    mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
                progress_bar.progress(60)

                # Resize mask ke ukuran asli-krn mau dilabeli
                mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                # Ambil nama file asli dari path
                original_filename = os.path.basename(image_path)  # Contoh: 'timestamp_123456.png'
                # Simpan hasil mask dengan nama yang sama
                cv2.imwrite(f"./segmen_images/{choice_video}/{original_filename}", mask_resized* 255) #output kn 0/1, terlalu gelap mkny dbalikin*255
                progress_bar.progress(80)

                # Labelling n bounding box pd mask
                labeled_mask = sklabel(mask_resized * 255)
                regions = regionprops(labeled_mask)
                result_image = np.array(image_rgb) #grayscale to rgb

                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)

                    if regions:
                        for region in regions:    
                            # Ambil nilai segmented img (min_row-top, min_col-left, max_row-bottom, max_col-right)
                            minr, minc, maxr, maxc = region.bbox
                            h = maxr - minr #brp px
                            w = maxc - minc 
                            x1, y1 = minc, minr
                            x2, y2 = maxc, maxr
                        
                            pixel_spacing_mm = 50/1048   #0.04 mm/px 
                            width_mm = w * pixel_spacing_mm
                            height_mm = h * pixel_spacing_mm
                            # Store measurements
                            arr_width_artery.append(width_mm)
                            arr_height_artery.append(height_mm)
                            arr_width_px.append(w)
                            arr_height_px.append(h)
                            # tulis hasil ke CSV
                            #Tulis hasil ke CSV: filename, width_mm, height_mm, width_px, height_px, x1, y1, x2, y2
                            writer.writerow([
                                original_filename,
                                round(width_mm, 2), round(height_mm, 2),
                                w, h,
                                x1, y1, x2, y2
                            ])
                            # Tambah label ke gambar
                            width_label = f"W: {w}px / {width_mm:.2f}mm"
                            height_label = f"H: {h}px / {height_mm:.2f}mm"
                            cv2.putText(result_image, width_label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(result_image, height_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Add rectangle
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else: 
                            # Jika tidak ada region terdeteksi tetap simpan baris kosong
                            writer.writerow([original_filename, "0", "0", "0", "0", "0", "0", "0", "0"])
                            expander.warning(f"Tidak ada region terdeteksi di frame {count}")
                # Simpan hasil citra (dengan atau tanpa bounding box)
                cv2.imwrite(f"./artery_height_images/{choice_video}/{original_filename}", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                progress_bar.progress(100)
                progress_segment.progress(int((count / total_images) * 100))   #proses keseluruhan         

        # Show Output Video
        img = cv2.imread(f"./artery_height_images/{choice_video}/{sorted(os.listdir(f'./artery_height_images/{choice_video}'))[0]}")
        # Tampilkan video jika sudah ada
        video_output_folder = (f"./video_output/{choice_video}")
        os.makedirs(video_output_folder, exist_ok=True)  # pastikan foldernya ada
        OUTPUT_FILE = os.path.join(video_output_folder, 'output_video.mp4')
        CONVERTED_FILE = os.path.join(video_output_folder, 'converted_output.mp4')
        if os.path.exists(CONVERTED_FILE):
            st.video(CONVERTED_FILE)

        show_output_video = st.button('Show Output Video')
        if show_output_video:         
            progress_video = st.progress(0) 
            height, width, channels = img.shape

            # output video configuration
            FPS = 31
            WIDTH = width
            HEIGHT = height
            # define video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT)) #dr frame ke video

            # Ambil semua nama file citra secara urut
            frame_folder = f"./artery_height_images/{choice_video}"
            frame_files = sorted(os.listdir(frame_folder))
            total_frames = len(frame_files)

            for i, filename in enumerate(frame_files):
                frame_path = os.path.join(frame_folder, filename)
                frame = cv2.imread(frame_path)

                if frame is not None:
                    writer.write(frame)
                        
                # Update progress
                progress_video.progress((i + 1) / total_frames) 

            writer.release()

            # Konversi ke format yang kompatibel (libx264)
            ffmpeg_path = r"D:\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

            subprocess.run([
                ffmpeg_path,
                '-y',
                '-i', OUTPUT_FILE,
                '-vcodec', 'libx264',
                '-crf', '23',
                CONVERTED_FILE
            ],check=True)
            st.video(CONVERTED_FILE)
        
        st.header('Graph Sensor')
        file = f"./raw_sensor/{choice_video}.csv"
        # Baca CSV
        df = pd.read_csv(file)
        # Konversi nilai sensor ke kiloPascal (area probe = 0.00045 m2, /1000 ke kpa)
        df["Pressure (kPa)"] = (df["Sensor Value"] /0.00045/1000).round(2)
        # Ganti nama kolom untuk tampilan
        df.columns = ["Timestamp (s)", "Sensor Value (N)", "Pressure (kPa)"]
        # Tampilkan tabel
        st.dataframe(df)
        
        #Exclude Outlier
        raw_csv   = f'./arheight_measurements/{choice_video}/artery_geometry.csv'
        clean_csv = f'./arheight_measurementsout/{choice_video}/artery_geometry.csv'
        if os.path.exists(raw_csv):
            df_art = pd.read_csv(raw_csv)

            zero_cnt = (df_art["Height (mm)"] == 0).sum()
            df_art.loc[df_art["Height (mm)"] == 0, "Height (mm)"] = np.nan

            # Pastikan folder tujuan ada
            os.makedirs(os.path.dirname(clean_csv), exist_ok=True)
            df_art.to_csv(clean_csv, index=False)

            st.write(f"{zero_cnt} nilai 'Height = 0' diganti NaN lalu disimpan ke _out")
        else:
            st.warning(f"File artery_geometry.csv mentah tidak ditemukan pada {choice_video}")

        st.header('Sensor X Vessel Radius')
        #Konversi Timestamp filename ke detik = buat time difference
        def timestamp_to_seconds(t_str):
            h, m, s, ms = map(int, t_str.split('-'))
            return round(h * 3600 + m * 60 + s + ms / 1000, 2)

        # Load Artery Height
        artery_file = clean_csv
        artery_available = os.path.exists(artery_file)
        # Load Anotasi Subject
        anot_file = f'./annotated/{choice_video}/data.csv'
        anot_available = os.path.exists(anot_file)
        # Load sensor data 
        sensor_file = f'./raw_sensor/{choice_video}.csv'
        sensor_available = os.path.exists(sensor_file)

        if artery_available and sensor_available:
            # BACA DATA ARTERI 
            artery_df = pd.read_csv(artery_file) #baca 
            if "Height (mm)" in artery_df.columns:
                y_radius = artery_df['Height (mm)'].tolist() #ambil height aja dr file 
            else:
                st.warning("'Height (mm)' tidak ditemukan dalam artery_geometry.csv")
                y_radius = []
            
            if 'Filename' not in artery_df.columns:
                image_dir = f"./artery_height_images/{choice_video}"
                if os.path.exists(image_dir):
                    image_filenames = sorted(os.listdir(image_dir))
                    image_filenames = natsort.natsorted(image_filenames)
                    artery_df['Filename'] = image_filenames[:len(artery_df)]
                else:
                    artery_df['Filename'] = [''] * len(artery_df)
                
            # BACA DATA SENSOR 
            sensor_df = pd.read_csv(sensor_file)
            sensor_df["Pressure (kPa)"] = (sensor_df["Sensor Value"] / 0.00045 / 1000).round(2)
            sensor_df.columns = ["Timestamp (s)", "Sensor Value (N)", "Pressure (kPa)"]
            y_sensor = sensor_df["Pressure (kPa)"].tolist() #yg diambil pressure
            min_sensor = min(y_sensor)
            y_sensor_visual = [val - min_sensor for val in y_sensor]
            original_filenames = sensor_df["Timestamp (s)"].tolist() #simpan timestamp asli sensor

            artery_df = artery_df.sort_values("Filename").reset_index(drop=True) #sort aja

            # Konversi t.sensor ke detik, anggap dimulai dr 0, sumbu x
            timestamps_raw = [timestamp_to_seconds(t) for t in original_filenames]
            timestamps = [t - timestamps_raw[0] for t in timestamps_raw]
            # Timestamp untuk radius (dari filename) ke detik
            y_radius = artery_df['Height (mm)'].tolist()
            timestamps_radius = [timestamp_to_seconds(f.replace(".png", "")) for f in artery_df["Filename"]]
            timestamps_radius = [t - timestamps_raw[0] for t in timestamps_radius]

            
            # Juml maks frame=y_radius (data tinggi arteri) ===
            max_frame = len(y_radius) - 1  # slider hanya untuk frame citraF

            # Ambil gambar untuk preview slider
            image_dir = f"./artery_height_images/{choice_video}"
            if os.path.exists(image_dir):
                image_filenames = sorted(os.listdir(image_dir)) #ambil
                image_filenames = natsort.natsorted(image_filenames) #sort
                image_filenames = image_filenames[:min(len(image_filenames), len(y_radius))] #sinkronin jmlh gambar n arheight
                max_frame = len(image_filenames) - 1  # updt maxframe biar slider ga out of index

            # === SLIDERS UNTUK MEMILIH FRAME ===
            st.subheader("Select point according to the ultrasound procedure:")
            slider_time_a = st.slider("**Point A (s) - Press**", float(timestamps[0]), float(timestamps[-1]), float(timestamps[0]), step=0.01)
            slider_time_b = st.slider("**Point B (s) - Release**", float(timestamps[0]), float(timestamps[-1]), float(timestamps[1]), step=0.01)

            # Cari indeks sensor A dan B
            idx_sensor_a = np.argmin(np.abs(np.array(timestamps) - slider_time_a))
            idx_sensor_b = np.argmin(np.abs(np.array(timestamps) - slider_time_b))

            # Cari frame radius yang paling dekat dengan waktu sensor
            idx_frame_a = np.argmin(np.abs(np.array(timestamps_radius) - slider_time_a))
            idx_frame_b = np.argmin(np.abs(np.array(timestamps_radius) - slider_time_b))

            # === GRAFIK KONTAK FORCE + ARTERY HEIGHT ==
            y_line = np.arange(0, max(y_sensor), 1)
            x2 = slider_time_a
            x3 = slider_time_b #tp emg dipake buat pilih frame
            fig, ax1 = plt.subplots() #kyk waktu frame A itu terjadi di xxx waktu sensor

            #grafik utama itu emg pake sensor krn pingin tahu kapan tekanan naik/turun, 
            #nnt citranya yg ngikuti, jgn kebalik.

            # Plot contact force (Pressure) di sumbu y kiri
            ax1.scatter(timestamps, y_sensor_visual, color='#d62728', s=15, zorder=2)
            ax1.axvline(x=x2, color='orange', linewidth=2.5, zorder=3)
            ax1.axvline(x=x3, color='green', linewidth=2.5, zorder=3)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylim(-1, 23)  # Sensor: -1 to 18 kPa
            ax1.set_ylabel('Contact Force (kPa)', color='#d62728')
            ax1.tick_params(axis='y', labelcolor='#d62728')

            lines1, labels1 = ax1.get_legend_handles_labels()

            # Checkbox untuk menampilkan grafik diameter
            show_diameter = st.checkbox("Show Artery Height (mm)", value=True)
            show_anot = st.checkbox("Show Annotation Data", value=False)

            # Buat axis kanan (selalu buat agar tidak error)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Artery Height / Annotation (mm)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.patch.set_visible(False)

            # Hanya tampilkan sumbu kanan jika user pilih checkbox
            #if show_diameter:
                #ax2 = ax1.twinx()
                #ax2.plot(timestamps_radius, y_radius, color='blue', label='Artery Height (mm)', linewidth=1)
                #ax2.set_ylabel('Artery Height (mm)', color='blue')
                #ax2.tick_params(axis='y', labelcolor='blue')
                #ax1.set_zorder(ax2.get_zorder() + 1)
                #ax1.patch.set_visible(False)

                # Gabungkan legend dari keduanya
                #lines2, labels2 = ax2.get_legend_handles_labels()
                #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            #else:
                # Hanya legend dari ax1
                #ax1.legend(lines1, labels1, loc='upper right')

            # Buat formatter dan skala kanan
            artery_scale_factor = 24 / 10
            ax2.set_ylim(-1, 23)
            def mm_formatter(x, pos):
                return f"{x / artery_scale_factor:.0f}"
            ax2.yaxis.set_major_formatter(FuncFormatter(mm_formatter))
            ax2.yaxis.set_major_locator(MultipleLocator(2 * artery_scale_factor))

            # Plot artery height jika dicentang
            if show_diameter:
                scaled_y_radius = [val * artery_scale_factor for val in y_radius]
                ax2.plot(timestamps_radius, scaled_y_radius, color='blue', linewidth=1)

            # Plot anotasi jika file tersedia dan dicentang
            if anot_available and show_anot:
                anot_df = pd.read_csv(anot_file)
                if 'filename' in anot_df.columns and 'height_mm' in anot_df.columns:
                    anot_df_filtered = anot_df[
                        (anot_df['height_mm'] != 0) &
                        (anot_df['filename'].astype(str).str.endswith(".png"))
                    ].copy()

                    anot_timestamps_raw = [timestamp_to_seconds(t.replace(".png", "")) for t in anot_df_filtered['filename']]
                    anot_timestamps = [t - timestamps_raw[0] for t in anot_timestamps_raw]
                    anot_values = anot_df_filtered['height_mm'].tolist()
                    scaled_anot_values = [val * artery_scale_factor for val in anot_values]

                    ax2.scatter(anot_timestamps, scaled_anot_values, color='purple', s=25, marker='x', zorder=5)
            
            # Gabungkan legend dari semua plot yang aktif
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        else:
            st.info("Data anotasi belum tersedia untuk video ini.")

        # === TAMPILKAN 3 GAMBAR BERDASARKAN NAMA FILE ASLI ===
        col1, col2 = st.columns([3, 1])
        filename_a = image_filenames[idx_frame_a]
        filename_b = image_filenames[idx_frame_b] 

        path_a = os.path.join(image_dir, filename_a)
        path_b = os.path.join(image_dir, filename_b)

        with col1:
            st.pyplot(fig)
            col_info1, col_info2 = st.columns(2)
            for label, t, idx_frame, idx_sensor, col in zip(
                ["A", "B"],
                [slider_time_a, slider_time_b],
                [idx_frame_a, idx_frame_b],
                [idx_sensor_a, idx_sensor_b],
                [col_info1, col_info2]
            ):
                radius_val = round(y_radius[idx_frame], 2)
                pressure_val = round(y_sensor[idx_sensor], 2)
                filename = image_filenames[idx_frame]

                with col:
                    st.markdown(f"**Point {label}**")
                    st.markdown(f"- File: **{filename}**")
                    st.markdown(f"- Timestamp: **{round(t, 3)} s**")
                    st.markdown(f"- Pressure: **{pressure_val} kPa**")
                    st.markdown(f"- Radius: **{radius_val} mm**")
    
        with col2:
            try:
                st.image(path_a, caption=f"File: {filename_a}")
                st.image(path_b, caption=f"File: {filename_b}")

            except Exception as e:
                st.error(f"Gagal memuat gambar: {e}") 
        
        st.header('Vascular Elasticity Characterization')
        st.subheader("📘 Young's Modulus")
        # Bagi jadi dua kolom
        col1, col2 = st.columns(2)
        # Kiri: Rumus dasar modulus Young
        with col1:
            st.latex(r"\text{Stress} = \frac{F}{A} \quad (\text{kPa})")
            st.latex(r"\text{Strain} = \frac{\Delta L}{L_0} \quad (\text{mm})")
            st.latex(r"\text{Elasticity} = \frac{\text{Stress}}{\text{Strain}} \quad (\text{kPa})")
        # Kanan: Rumus pengolahan data radius dan sensor
        with col2:
            st.latex(r"\text{Contact Pressure Changes} = \text{Pressure Max} - \text{Pressure Min}")
            st.latex(r"\text{Artery Diameter Changes} = \frac{\text{Min Pressure Height} - \text{Max Pressure Height}}{\text{Min Pressure Height}}")
            st.latex(r"\text{Elasticity} = \frac{\text{Contact Pressure Changes}}{\text{Artery Diameter Changes}}")
        st.markdown("---")

        # === HITUNG PARAMETER ELASTISITAS BERDASARKAN FRAME TERPILIH ===
        pressure_a = y_sensor[idx_sensor_a]
        pressure_b = y_sensor[idx_sensor_b]

        radius_a = y_radius[idx_frame_a]
        radius_b = y_radius[idx_frame_b]

        timestamp_a = timestamps[idx_sensor_a]
        timestamp_b = timestamps[idx_sensor_b]

        # 1. Contact Pressure Change
        contact_pressure_change = round(abs(pressure_a - pressure_b), 3)

        # 2. Artery Diameter Change (pakai rumus proporsional)
        if radius_b != 0:
            artery_diameter_change = round(abs((radius_b - radius_a) / radius_b), 5)
        else:
            artery_diameter_change = float('inf')

        # 3. Elasticity
        if artery_diameter_change != 0:
            elasticity = round(contact_pressure_change / artery_diameter_change, 3)
        else:
            elasticity = float('inf')

        # 4. Time difference
        time_diff = round(abs(timestamp_b - timestamp_a), 3)

        # === TAMPILKAN HASILNYA
        st.subheader("🔍 Elasticity Parameter")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🧪 Contact Pressure Change", f"{contact_pressure_change} kPa")
            st.metric("📏 Artery Diameter Change", f"{artery_diameter_change}")

        with col2:
            st.metric("💡 Elasticity", f"{elasticity} kPa")
            st.metric("⏱️ Time Difference", f"{time_diff} s")

    elif choice == "Image Processing":
        st.title("Image Processing Visualization")
        st.header("Image Pre-processing")
        image_paths = sorted(glob.glob("./raw_images/Subject1/*.png"))
        
        if image_paths:
            sample_path = image_paths[0]  # Ambil satu gambar sebagai contoh
            image_rgb = Image.open(sample_path).convert("RGB")
            image_size = (384, 384)
            image_resized = image_rgb.resize(image_size)
            image_gray = image_resized.convert("L")
            
            # Simpan array untuk histogram
            rgb_np = np.array(image_rgb)
            resized_np = np.array(image_resized)
            gray_np = np.array(image_gray)
            
            # Normalisasi manual (ToTensor + Normalize)
            tensor_image = transforms.ToTensor()(image_gray)
            norm_image = transforms.Normalize(mean=[0.5], std=[0.5])(tensor_image) #krn grayscale kan cuma 1
            norm_np = norm_image.squeeze().numpy()

            # Layout Visualisasi
            st.subheader("1. Input Image (RGB)")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Input Image")
                st.markdown(f"- **Width x Height:** {image_rgb.size[0]} x {image_rgb.size[1]}")
                st.markdown(f"- **Format:** {image_rgb.format or 'PNG'}")
                st.markdown(f"- **Colour Mode:** {image_rgb.mode}")
                st.markdown(f"- **Data Type (dtype):** {rgb_np.dtype}")
                st.markdown(f"- **Pixel Value:** {rgb_np.min()} - {rgb_np.max()}")
            with col2:
                fig_r, ax_r = plt.subplots()
                ax_r.hist(rgb_np[:, :, 0].ravel(), bins=256, color='red', alpha=0.8)
                ax_r.set_title("Red Channel")
                ax_r.set_xlim(0, 255)
                ax_r.set_xlabel("Intensitas")
                st.pyplot(fig_r)

                fig_g, ax_g = plt.subplots()
                ax_g.hist(rgb_np[:, :, 1].ravel(), bins=256, color='green', alpha=0.8)
                ax_g.set_title("Green Channel")
                ax_g.set_xlim(0, 255)
                ax_g.set_xlabel("Intensitas")
                st.pyplot(fig_g)

                fig_b, ax_b = plt.subplots()
                ax_b.hist(rgb_np[:, :, 2].ravel(), bins=256, color='blue', alpha=0.8)
                ax_b.set_title("Blue Channel")
                ax_b.set_xlim(0, 255)
                ax_b.set_xlabel("Intensitas")
                st.pyplot(fig_b)

            st.subheader("2. Resize (384x384)")
            col3, col4 = st.columns(2)
            with col3:
                st.image(image_resized, caption="Resized Image")
            with col4:
                st.markdown("3 Channel: Red, Green, Blue")
                fig2, ax2 = plt.subplots()
                ax2.hist(resized_np[:, :, 0].ravel(), bins=256, color='red', alpha=0.5, label='Red')
                ax2.hist(resized_np[:, :, 1].ravel(), bins=256, color='green', alpha=0.5, label='Green')
                ax2.hist(resized_np[:, :, 2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
                ax2.set_title("Histogram RGB")
                ax2.set_xlim(0, 255)
                ax2.set_xlabel("Pixel Intensity")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                st.pyplot(fig2)

            st.subheader("3. Grayscale")
            col5, col6 = st.columns(2)
            with col5:
                st.image(image_gray, caption="Grayscale Image")
            with col6:
                st.markdown("1 Channel: Grayscale")
                fig3, ax3 = plt.subplots()
                ax3.hist(gray_np.ravel(), bins=256, color='gray', alpha=0.7)
                ax3.set_title("Histogram Grayscale")
                st.pyplot(fig3)

            st.subheader("4. To Tensor")
            col7, col8 = st.columns(2)
            with col7:
                # Ambil data pixel sebelum ToTensor (uint8)
                st.markdown("**Grayscale Pixel Value (0–255):**")
                fig_gray_val, ax_gray_val = plt.subplots()
                im = ax_gray_val.imshow(gray_np, cmap='gray', vmin=0, vmax=255)
                fig_gray_val.colorbar(im, ax=ax_gray_val)
                ax_gray_val.set_title("Grayscale Pixel Value (0–255)")
                st.pyplot(fig_gray_val)
            with col8:
            # Visualisasi nilai setelah ToTensor (float, 0–1)
                st.markdown("**Tensor Pixel Value (0–1):**")
                tensor_np = tensor_image.squeeze().numpy()
                fig_tensor_val, ax_tensor_val = plt.subplots()
                im2 = ax_tensor_val.imshow(tensor_np, cmap='gray', vmin=0, vmax=1)
                fig_tensor_val.colorbar(im2, ax=ax_tensor_val)
                ax_tensor_val.set_title("Grayscale Pixel Value (0–1)")
                st.pyplot(fig_tensor_val)

            st.subheader("5. Normalisasi (Mean=0.5, Std=0.5)")
            st.latex(r"x_{\text{norm}} = \frac{x - \mu}{\sigma}")

            st.markdown("""
            **Keterangan:**
            - \(x\) : Nilai piksel setelah `ToTensor()` (rentang 0–1)
            - \(\mu = 0.5\) : Nilai rata-rata (mean)
            - \(\sigma = 0.5\) : Standar deviasi (std)
            - \(x_{\\text{norm}}\) : Nilai setelah normalisasi (rentang ~ -1 s.d. 1)
            """)

            st.markdown("### Contoh Perhitungan:")
            st.latex(r"\text{Jika } x = 0 \Rightarrow x_{\text{norm}} = \frac{0 - 0.5}{0.5} = -1")
            st.latex(r"\text{Jika } x = 0.5 \Rightarrow x_{\text{norm}} = \frac{0.5 - 0.5}{0.5} = 0")
            st.latex(r"\text{Jika } x = 1.0 \Rightarrow x_{\text{norm}} = \frac{1.0 - 0.5}{0.5} = 1")

        st.header("Image Segmentation")
        st.subheader("6. Image Segmentation")
        col9, col10 = st.columns(2)
        # Load image segmentasi
        seg_path = "./our_team/segmen192x192.png"
        if os.path.exists(seg_path):
            seg_image = Image.open(seg_path)
            seg_np = np.array(seg_image)
            seg_image = Image.open(seg_path).convert("L")  # Pastikan grayscale
            seg_np = np.array(seg_image)

            with col9:
                st.image(seg_image, caption="Segmented Image")
                st.markdown(f"- **Width x Height:** {seg_image.size[0]} x {seg_image.size[1]}")
                st.markdown(f"- **Format:** {seg_image.format or 'PNG'}")
                st.markdown(f"- **Colour Mode:** {seg_image.mode}")
                st.markdown(f"- **Data Type (dtype):** {seg_np.dtype}")
                st.markdown(f"- **Pixel Value Range:** {seg_np.min()} - {seg_np.max()}")
            with col10:
                fig_seg, ax_seg = plt.subplots()
                ax_seg.hist(seg_np.ravel(), bins=[0, 1, 255, 256], color='blue', rwidth=0.6)
                ax_seg.set_xticks([0, 255])
                ax_seg.set_title("Histogram Segmentasi (0 dan 255)")
                ax_seg.set_xlabel("Nilai Piksel")
                ax_seg.set_ylabel("Frekuensi")
                st.pyplot(fig_seg)
            
            st.subheader("7. Resize Image Segmentation to Original Size")
            col11, col12 = st.columns(2)
            seg2_path = "./our_team/segmented.png"
            if os.path.exists(seg_path):
                seg2_image = Image.open(seg2_path)
                seg2_np = np.array(seg2_image)
                seg2_image = Image.open(seg2_path).convert("L")  # Pastikan grayscale
                seg2_np = np.array(seg2_image)
                with col11:
                    st.image(seg2_image, caption="Segmented Image Resize to Original Size ")
                    st.markdown(f"- **Width x Height:** {seg2_image.size[0]} x {seg2_image.size[1]}")
                    st.markdown(f"- **Format:** {seg2_image.format or 'PNG'}")
                    st.markdown(f"- **Colour Mode:** {seg2_image.mode}")
                    st.markdown(f"- **Data Type (dtype):** {seg2_np.dtype}")
                    st.markdown(f"- **Pixel Value Range:** {seg2_np.min()} - {seg2_np.max()}")
                with col12:
                    fig_seg, ax_seg = plt.subplots()
                    ax_seg.hist(seg2_np.ravel(), bins=[0, 1, 255, 256], color='blue', rwidth=0.6)
                    ax_seg.set_xticks([0, 255])
                    ax_seg.set_title("Histogram Segmentasi (0 dan 255)")
                    ax_seg.set_xlabel("Nilai Piksel")
                    ax_seg.set_ylabel("Frekuensi")
                    st.pyplot(fig_seg)
            
            st.subheader("8. Labelling with Region Properties")
            # Load image dari path
            lbl_path = "./our_team/labelimg.png"
            image_pil = Image.open(lbl_path).convert("RGB")  # pastikan 3 channel
            rgb_np = np.array(image_pil)  # konversi ke numpy array

            # Buat dua kolom
            col13, col14 = st.columns(2)

            with col13:
                st.image(image_pil, caption="Labeled Image")
                st.markdown(f"- **Width x Height:** {image_pil.size[0]} x {image_pil.size[1]}")
                st.markdown(f"- **Format:** {image_pil.format or 'PNG'}")
                st.markdown(f"- **Colour Mode:** {image_pil.mode}")
                st.markdown(f"- **Data Type (dtype):** {rgb_np.dtype}")
                st.markdown(f"- **Pixel Value:** {rgb_np.min()} - {rgb_np.max()}")

            with col14:
                if rgb_np.ndim == 3 and rgb_np.shape[2] == 3:
                    # Pisah channel R, G, B
                    R, G, B = rgb_np[:, :, 0], rgb_np[:, :, 1], rgb_np[:, :, 2]

                    # Plot histogram
                    fig, ax = plt.subplots()
                    ax.hist(R.ravel(), bins=256, color='red', alpha=0.5, label='Red')
                    ax.hist(G.ravel(), bins=256, color='green', alpha=0.5, label='Green')
                    ax.hist(B.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')

                    ax.set_title("RGB Histogram")
                    ax.set_xlabel("Pixel Intensity")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("Gambar tidak memiliki 3 channel warna.")


            



                

if __name__ == "__main__":
    main()


