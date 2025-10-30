
import streamlit as st
import os
import numpy as np
import pydicom
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import mahalanobis
from PIL import Image
import torch
from torchvision import transforms
from torch_fidelity import calculate_metrics
from tempfile import TemporaryDirectory
from sewar.full_ref import vifp
import warnings
import zipfile
import io

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Image Quality Metrics Calculator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Display logo/header image
try:
    logo = Image.open('TPU_yadernikh.jpg')
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(logo, use_container_width=True)
except:
    st.warning("⚠️ Logo file not found. Please ensure TPU_yadernikh.jpg is in the same directory.")

st.markdown('<p class="main-header">🔬 Image Quality Metrics Calculator</p>', unsafe_allow_html=True)
st.markdown("### Medical Image Quality Metrics Evaluation (IMA/DICOM)")

# Sidebar for instructions
with st.sidebar:
    st.header("📖 Usage Guide")
    st.markdown("""
    **Steps:**
    1. Upload a ZIP folder containing IMA/DICOM files
    2. Pilih metrik yang ingin dihitung
    3. Klik tombol "Hitung Metrik"
    4. Unduh hasil perhitungan

    **Format File:**
    - Format: .IMA atau .dcm (DICOM)
    - Folder: Clean, Noisy, Denoised
    - Pastikan jumlah file sama di setiap folder

    **Metrik yang tersedia:**
    - MSE (Mean Squared Error)
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - Mahalanobis Distance
    - FID (Fréchet Inception Distance)
    - VIF (Visual Information Fidelity)
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "⚙️ Pengaturan", "📊 Hasil"])

with tab1:
    st.subheader("Upload Folder Citra")
    col1, col2, col3 = st.columns(3)

    with col1:
        clean_zip = st.file_uploader(
            "Upload Clean Images (ZIP)",
            type=['zip'],
            help="Upload file ZIP berisi citra clean"
        )

    with col2:
        noisy_zip = st.file_uploader(
            "Upload Noisy Images (ZIP)",
            type=['zip'],
            help="Upload file ZIP berisi citra noisy"
        )

    with col3:
        denoised_zip = st.file_uploader(
            "Upload Denoised Images (ZIP)",
            type=['zip'],
            help="Upload file ZIP berisi citra denoised"
        )

with tab2:
    st.subheader("Pilih Metrik yang Ingin Dihitung")
    col1, col2 = st.columns(2)

    with col1:
        calc_mse = st.checkbox("MSE (Mean Squared Error)", value=True)
        calc_psnr = st.checkbox("PSNR (Peak Signal-to-Noise Ratio)", value=True)
        calc_ssim = st.checkbox("SSIM (Structural Similarity)", value=True)

    with col2:
        calc_mahalanobis = st.checkbox("Mahalanobis Distance", value=False)
        calc_fid = st.checkbox("FID (Fréchet Inception Distance)", value=False)
        calc_vif = st.checkbox("VIF (Visual Information Fidelity)", value=False)

    st.info("⚠️ Catatan: FID memerlukan waktu komputasi yang lama dan memori yang besar")

    if calc_mahalanobis:
        patch_size = st.slider("Ukuran Patch untuk Mahalanobis Distance", 8, 64, 32, 8)
    else:
        patch_size = 32

# Helper functions
@st.cache_data
def extract_zip(zip_file):
    """Extract ZIP file and return temporary directory path"""
    import tempfile
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

@st.cache_data
def load_dicom_images(folder_path):
    """Load DICOM/IMA images from folder"""
    images = []
    filenames = []

    # Find IMA/DCM files recursively
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.ima', '.dcm')):
                all_files.append(os.path.join(root, file))

    all_files = sorted(all_files)

    for file_path in all_files:
        try:
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            images.append(img)
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            st.warning(f"Error loading {os.path.basename(file_path)}: {e}")
            continue

    return images, filenames

def calculate_mse_psnr_ssim(clean_imgs, noisy_imgs, denoised_imgs, filenames):
    """Calculate MSE, PSNR, and SSIM metrics"""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (clean, noisy, denoised, fname) in enumerate(zip(clean_imgs, noisy_imgs, denoised_imgs, filenames)):
        try:
            if clean.shape != noisy.shape or clean.shape != denoised.shape:
                continue

            result = {'Filename': fname}

            if calc_mse:
                mse_noisy = mean_squared_error(clean.flatten(), noisy.flatten())
                mse_denoised = mean_squared_error(clean.flatten(), denoised.flatten())
                result['MSE Noisy'] = mse_noisy
                result['MSE Denoised'] = mse_denoised

            if calc_psnr:
                data_range = clean.max() - clean.min()
                psnr_noisy = psnr(clean, noisy, data_range=data_range)
                psnr_denoised = psnr(clean, denoised, data_range=data_range)
                result['PSNR Noisy'] = psnr_noisy
                result['PSNR Denoised'] = psnr_denoised

            if calc_ssim:
                data_range = clean.max() - clean.min()
                ssim_noisy = ssim(clean, noisy, data_range=data_range)
                ssim_denoised = ssim(clean, denoised, data_range=data_range)
                result['SSIM Noisy'] = ssim_noisy
                result['SSIM Denoised'] = ssim_denoised

            results.append(result)

            progress = (idx + 1) / len(clean_imgs)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx + 1}/{len(clean_imgs)} images")

        except Exception as e:
            st.warning(f"Error processing {fname}: {e}")
            continue

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)

def extract_patches(img, patch_size):
    """Extract non-overlapping patches"""
    patches = []
    h, w = img.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch.flatten())
    return np.array(patches)

def calculate_mahalanobis_distance(clean_imgs, noisy_imgs, denoised_imgs, filenames, patch_size):
    """Calculate Mahalanobis Distance"""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (clean, noisy, denoised, fname) in enumerate(zip(clean_imgs, noisy_imgs, denoised_imgs, filenames)):
        try:
            clean_patches = extract_patches(clean, patch_size)
            noisy_patches = extract_patches(noisy, patch_size)
            denoised_patches = extract_patches(denoised, patch_size)

            mean_clean = np.mean(clean_patches, axis=0)
            cov_clean = np.cov(clean_patches, rowvar=False)
            cov_clean += np.eye(cov_clean.shape[0]) * 1e-5
            inv_cov = np.linalg.pinv(cov_clean)

            md_noisy = []
            md_denoised = []

            for n_patch, d_patch in zip(noisy_patches, denoised_patches):
                try:
                    d1 = mahalanobis(n_patch, mean_clean, inv_cov)
                    d2 = mahalanobis(d_patch, mean_clean, inv_cov)
                    if not np.isnan(d1):
                        md_noisy.append(d1)
                    if not np.isnan(d2):
                        md_denoised.append(d2)
                except:
                    continue

            if md_noisy and md_denoised:
                results.append({
                    'Filename': fname,
                    'Mean Mahalanobis Noisy-Clean': np.mean(md_noisy),
                    'Mean Mahalanobis Denoised-Clean': np.mean(md_denoised)
                })

            progress = (idx + 1) / len(clean_imgs)
            progress_bar.progress(progress)
            status_text.text(f"Processing Mahalanobis: {idx + 1}/{len(clean_imgs)} images")

        except Exception as e:
            st.warning(f"Error processing Mahalanobis for {fname}: {e}")
            continue

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)

def calculate_fid_metric(clean_imgs, noisy_imgs, denoised_imgs):
    """Calculate FID metric"""
    with st.spinner("Calculating FID (this may take a while)..."):
        try:
            with TemporaryDirectory() as temp_clean_dir, \
                 TemporaryDirectory() as temp_noisy_dir, \
                 TemporaryDirectory() as temp_denoised_dir:

                # Save images as PNG
                for i, img in enumerate(clean_imgs):
                    img_norm = (img - img.min()) / (img.max() - img.min())
                    pil_img = Image.fromarray((img_norm * 255).astype(np.uint8))
                    pil_img.save(os.path.join(temp_clean_dir, f'image_{i}.png'))

                for i, img in enumerate(noisy_imgs):
                    img_norm = (img - img.min()) / (img.max() - img.min())
                    pil_img = Image.fromarray((img_norm * 255).astype(np.uint8))
                    pil_img.save(os.path.join(temp_noisy_dir, f'image_{i}.png'))

                for i, img in enumerate(denoised_imgs):
                    img_norm = (img - img.min()) / (img.max() - img.min())
                    pil_img = Image.fromarray((img_norm * 255).astype(np.uint8))
                    pil_img.save(os.path.join(temp_denoised_dir, f'image_{i}.png'))

                # Calculate FID
                metrics_denoised = calculate_metrics(
                    input1=temp_clean_dir,
                    input2=temp_denoised_dir,
                    cuda=torch.cuda.is_available(),
                    isc=False,
                    fid=True,
                    kid=False,
                    verbose=False
                )

                metrics_noisy = calculate_metrics(
                    input1=temp_clean_dir,
                    input2=temp_noisy_dir,
                    cuda=torch.cuda.is_available(),
                    isc=False,
                    fid=True,
                    kid=False,
                    verbose=False
                )

                return {
                    'FID Clean-Denoised': metrics_denoised['frechet_inception_distance'],
                    'FID Clean-Noisy': metrics_noisy['frechet_inception_distance']
                }
        except Exception as e:
            st.error(f"Error calculating FID: {e}")
            return None

def calculate_vif_metric(clean_imgs, noisy_imgs, denoised_imgs, filenames):
    """Calculate VIF metric"""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (clean, noisy, denoised, fname) in enumerate(zip(clean_imgs, noisy_imgs, denoised_imgs, filenames)):
        try:
            # Normalize to [0, 1]
            clean_norm = (clean - clean.min()) / (clean.max() - clean.min())
            noisy_norm = (noisy - noisy.min()) / (noisy.max() - noisy.min())
            denoised_norm = (denoised - denoised.min()) / (denoised.max() - denoised.min())

            vif_noisy = vifp(clean_norm, noisy_norm, sigma_nsq=2.0)
            vif_denoised = vifp(clean_norm, denoised_norm, sigma_nsq=2.0)

            results.append({
                'Filename': fname,
                'VIF Noisy': vif_noisy,
                'VIF Denoised': vif_denoised
            })

            progress = (idx + 1) / len(clean_imgs)
            progress_bar.progress(progress)
            status_text.text(f"Processing VIF: {idx + 1}/{len(clean_imgs)} images")

        except Exception as e:
            st.warning(f"Error calculating VIF for {fname}: {e}")
            continue

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)

# Main calculation
with tab3:
    if st.button("🚀 Hitung Metrik", type="primary", use_container_width=True):
        if not (clean_zip and noisy_zip and denoised_zip):
            st.error("⚠️ Mohon upload semua folder (Clean, Noisy, Denoised)")
        else:
            try:
                st.info("📂 Extracting ZIP files...")
                clean_dir = extract_zip(clean_zip)
                noisy_dir = extract_zip(noisy_zip)
                denoised_dir = extract_zip(denoised_zip)

                st.info("🖼️ Loading images...")
                clean_imgs, clean_files = load_dicom_images(clean_dir)
                noisy_imgs, noisy_files = load_dicom_images(noisy_dir)
                denoised_imgs, denoised_files = load_dicom_images(denoised_dir)

                if len(clean_imgs) == 0 or len(noisy_imgs) == 0 or len(denoised_imgs) == 0:
                    st.error("❌ Tidak ada file IMA/DICOM yang valid ditemukan")
                elif len(clean_imgs) != len(noisy_imgs) or len(clean_imgs) != len(denoised_imgs):
                    st.error(f"❌ Jumlah file tidak sama: Clean={len(clean_imgs)}, Noisy={len(noisy_imgs)}, Denoised={len(denoised_imgs)}")
                else:
                    st.success(f"✅ Berhasil memuat {len(clean_imgs)} gambar dari setiap folder")

                    all_results = {}

                    # Calculate MSE, PSNR, SSIM
                    if calc_mse or calc_psnr or calc_ssim:
                        st.info("📊 Menghitung MSE, PSNR, SSIM...")
                        df_basic = calculate_mse_psnr_ssim(clean_imgs, noisy_imgs, denoised_imgs, clean_files)
                        all_results['Basic Metrics'] = df_basic
                        st.success("✅ MSE, PSNR, SSIM selesai dihitung")
                        st.dataframe(df_basic, use_container_width=True)

                    # Calculate Mahalanobis Distance
                    if calc_mahalanobis:
                        st.info("📊 Menghitung Mahalanobis Distance...")
                        df_mahal = calculate_mahalanobis_distance(clean_imgs, noisy_imgs, denoised_imgs, clean_files, patch_size)
                        all_results['Mahalanobis Distance'] = df_mahal
                        st.success("✅ Mahalanobis Distance selesai dihitung")
                        st.dataframe(df_mahal, use_container_width=True)

                    # Calculate FID
                    if calc_fid:
                        fid_results = calculate_fid_metric(clean_imgs, noisy_imgs, denoised_imgs)
                        if fid_results:
                            df_fid = pd.DataFrame([fid_results])
                            all_results['FID'] = df_fid
                            st.success("✅ FID selesai dihitung")
                            st.dataframe(df_fid, use_container_width=True)

                    # Calculate VIF
                    if calc_vif:
                        st.info("📊 Menghitung VIF...")
                        df_vif = calculate_vif_metric(clean_imgs, noisy_imgs, denoised_imgs, clean_files)
                        all_results['VIF'] = df_vif
                        st.success("✅ VIF selesai dihitung")
                        st.dataframe(df_vif, use_container_width=True)

                    # Save results
                    st.subheader("💾 Download Hasil")

                    for metric_name, df in all_results.items():
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {metric_name} (CSV)",
                            data=csv,
                            file_name=f"{metric_name.replace(' ', '_').lower()}_results.csv",
                            mime="text/csv"
                        )

                    st.success("🎉 Semua perhitungan selesai!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

st.markdown("---")
st.markdown("📝 **Catatan:** Aplikasi ini menggunakan file IMA/DICOM untuk perhitungan metrik kualitas citra medis.")
st.markdown("🏫 **Tomsk Polytechnic University** - School of Nuclear Technologies")
