import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import csv
import math
import os
import tempfile
import mimetypes
from typing import Optional, Tuple, List
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calcule l'angle en degrés entre trois points."""
    try:
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'angle: {e}")
        return 0.0

def smooth_angle(angles: deque, window_size: int = 5) -> float:
    """Applique un filtre de moyenne mobile pour lisser les angles."""
    if len(angles) == 0:
        return 0.0
    return sum(angles) / min(len(angles), window_size)

def predict_direction(angle: float) -> Tuple[str, float]:
    """Prédit la direction du tir et calcule un score de confiance."""
    if angle < 145:
        direction = "Gauche"
        confidence = 1.0 - (angle / 145) * 0.2
    elif 145 <= angle <= 175:
        direction = "Centre"
        confidence = 1.0 - (abs(angle - 160) / 15) * 0.2
    else:
        direction = "Droite"
        confidence = (angle - 175) / 35 if angle < 210 else 1.0
    return direction, min(max(confidence, 0.0), 1.0)

def draw_curved_arrow(frame: np.ndarray, center: Tuple[int, int], direction: str, radius: int = 60, frame_count: int = 0):
    """Dessine une flèche courbe avec un effet de pulsation."""
    # Effet de pulsation : le rayon varie légèrement
    pulse = int(5 * math.sin(frame_count * 0.1))  # Pulsation lente
    adjusted_radius = radius + pulse
    
    start_angle = 0
    end_angle = 0
    color = (0, 255, 0)  # Vert vif pour le terrain
    if direction == "Gauche":
        start_angle, end_angle = 0, 90
        color = (255, 165, 0)  # Orange pour Gauche
    elif direction == "Centre":
        start_angle, end_angle = -45, 45
        color = (255, 255, 255)  # Blanc pour Centre
    elif direction == "Droite":
        start_angle, end_angle = -90, 0
        color = (255, 0, 0)  # Rouge pour Droite
    
    cv2.ellipse(frame, center, (adjusted_radius, adjusted_radius), 0, start_angle, end_angle, color, 3)
    angle_rad = math.radians(end_angle)
    arrow_tip = (int(center[0] + adjusted_radius * math.cos(angle_rad)), int(center[1] + adjusted_radius * math.sin(angle_rad)))
    cv2.arrowedLine(frame, center, arrow_tip, color, 3, tipLength=0.3)

def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], font_scale: float = 1, text_color: Tuple[int, int, int] = (255, 255, 255), bg_color: Tuple[int, int, int] = (0, 128, 0)):
    """Dessine du texte avec un fond rectangulaire pour meilleure lisibilité."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10
    bg_top_left = (position[0] - padding, position[1] - text_height - padding)
    bg_bottom_right = (position[0] + text_width + padding, position[1] + padding)
    cv2.rectangle(frame, bg_top_left, bg_bottom_right, bg_color, -1)  # Fond vert foncé
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)

def detect_ball(frame: np.ndarray) -> bool:
    """Détecte un ballon de football dans l'image à l'aide de Hough Circle Transform."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)  # Cercle jaune
                # Ajouter un emoji ballon (simulé par un cercle blanc avec bordure noire)
                cv2.circle(frame, (x + r + 10, y), 5, (255, 255, 255), -1)
                cv2.circle(frame, (x + r + 10, y), 5, (0, 0, 0), 1)
            logger.info("Ballon détecté")
            return True
        logger.warning("Aucun ballon détecté")
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la détection du ballon: {e}")
        return False

def is_kicking_player(landmarks, width: int, height: int) -> Tuple[bool, str, np.ndarray, np.ndarray, np.ndarray]:
    """Vérifie si le joueur est en position de tirer (pas un gardien)."""
    try:
        left_hip = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x * width,
                             landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y * height])
        left_knee = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x * width,
                              landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y * height])
        left_ankle = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x * width,
                               landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y * height])
        
        right_hip = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x * width,
                              landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y * height])
        right_knee = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x * width,
                               landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y * height])
        right_ankle = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x * width,
                                landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y * height])
        
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        hip_y = (left_hip[1] + right_hip[1]) / 2
        if hip_y < height * 0.5:
            logger.warning("Joueur trop haut dans l'image (possible gardien)")
            return False, "Aucun", np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
        
        if left_angle < 160 or right_angle < 160:
            kicking_leg = "Gauche" if left_angle < right_angle else "Droite"
            hip = left_hip if kicking_leg == "Gauche" else right_hip
            knee = left_knee if kicking_leg == "Gauche" else right_knee
            ankle = left_ankle if kicking_leg == "Gauche" else right_ankle
            logger.info(f"Joueur en position de tir, jambe: {kicking_leg}")
            return True, kicking_leg, hip, knee, ankle
        else:
            logger.warning("Aucune jambe pliée (pas en position de tir)")
            return False, "Aucun", np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du joueur: {e}")
        return False, "Aucun", np.array([0, 0]), np.array([0, 0]), np.array([0, 0])

def validate_video_file(uploaded_file) -> bool:
    """Valide le fichier vidéo téléchargé."""
    if uploaded_file is None:
        return False
    
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if not mime_type or not mime_type.startswith('video'):
        st.error("Veuillez téléverser un fichier vidéo valide (ex: mp4, avi).")
        return False
    
    max_size = 200 * 1024 * 1024  # 200MB
    if uploaded_file.size > max_size:
        st.error("Le fichier est trop volumineux. La taille maximale est de 200MB.")
        return False
    
    return True

def process_video(input_path: str, output_path: str, csv_path: str, status_container) -> bool:
    """Traite la vidéo, détecte le joueur, le ballon, calcule les angles et génère les sorties."""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error(f"Erreur: Impossible d'ouvrir la vidéo {input_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        status_container.write(f"Propriétés de la vidéo: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error(f"Erreur: Impossible de créer la vidéo de sortie {output_path}")
            cap.release()
            return False

        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Frame', 'Ball_Detected', 'Is_Kick', 'Kicking_Leg', 'Angle', 'Direction', 'Confidence'])

            frame_count = 0
            angle_buffer = deque(maxlen=5)
            progress_bar = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                angle = 0.0
                direction = "Non détecté"
                confidence = 0.0
                kicking_leg = "Non détecté"
                is_kick = False
                ball_detected = detect_ball(frame)

                if results.pose_landmarks and ball_detected:
                    logger.info(f"Frame {frame_count}: Pose landmarks détectés")
                    is_kicker, kicking_leg, hip, knee, ankle = is_kicking_player(results.pose_landmarks.landmark, width, height)
                    
                    if is_kicker:
                        is_kick = True
                        try:
                            angle = calculate_angle(hip, knee, ankle)
                            angle_buffer.append(angle)
                            smoothed_angle = smooth_angle(angle_buffer)
                            direction, confidence = predict_direction(smoothed_angle)

                            mp_drawing.draw_landmarks(
                                frame, 
                                results.pose_landmarks, 
                                mp_holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0) if kicking_leg == "Gauche" else (0, 0, 255), thickness=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                            )

                            draw_text_with_background(frame, f'Jambe: {kicking_leg}', (50, 50), text_color=(255, 255, 255), bg_color=(0, 128, 0))
                            draw_text_with_background(frame, f'Angle: {smoothed_angle:.2f}°', (50, 100), text_color=(255, 255, 255), bg_color=(0, 128, 0))
                            draw_text_with_background(frame, f'Direction: {direction} ({confidence:.2%})', (50, 150), text_color=(255, 255, 255), bg_color=(0, 128, 0))
                            
                            arrow_center = (int(knee[0]), int(knee[1]))
                            draw_curved_arrow(frame, arrow_center, direction, frame_count=frame_count)
                        except Exception as e:
                            logger.warning(f"Erreur lors du traitement de la frame {frame_count}: {e}")
                            draw_text_with_background(frame, "Erreur de traitement", (50, 200), text_color=(255, 255, 255), bg_color=(0, 0, 128))
                    else:
                        draw_text_with_background(frame, "Ne pas un coup franc", (50, 50), text_color=(255, 255, 255), bg_color=(0, 0, 128))
                else:
                    logger.warning(f"Frame {frame_count}: Aucun coup franc détecté (ballon: {ball_detected}, landmarks: {bool(results.pose_landmarks)})")
                    draw_text_with_background(frame, "Ne pas un coup franc", (50, 50), text_color=(255, 255, 255), bg_color=(0, 0, 128))

                csv_writer.writerow([frame_count, str(ball_detected), str(is_kick), kicking_leg, f"{angle:.2f}", direction, f"{confidence:.2%}"])
                out.write(frame)
                progress_bar.progress(min(frame_count / total_frames, 1.0))

        status_container.write("Traitement terminé avec succès. ⚽")
        cap.release()
        out.release()
        return True

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la vidéo: {e}")
        st.error(f"Erreur lors du traitement de la vidéo: {e}")
        return False
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

def main():
    """Fonction principale pour l'interface Streamlit."""
    st.set_page_config(page_title="Détection de Coup Franc", layout="wide")

    # Appliquer un style CSS personnalisé
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1508098682722-8b951bd3382b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
            background-size: cover;
            background-attachment: fixed;
            color: white;
        }
        .stApp {
            background: rgba(0, 0, 0, 0.5);  /* Fond semi-transparent pour lisibilité */
        }
        h1 {
            color: #FFFFFF;
            text-shadow: 2px 2px 4px #000000;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #00FF00;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #00CC00;
        }
        .stProgress .st-bo {
            background-color: #FFFFFF;
        }
        .stProgress .st-bo > div {
            background-color: #00FF00;
        }
        .stDownloadButton>button {
            background-color: #00FF00;
            color: white;
            border-radius: 10px;
        }
        .stDownloadButton>button:hover {
            background-color: #00CC00;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Détection de la Direction d'un Coup Franc ⚽")
    st.markdown("""
        Téléchargez une vidéo de coup franc pour analyser la direction du tir basée sur l'angle de la jambe de frappe.  
        La vidéo doit être au format mp4 ou avi, ne pas dépasser 200MB, et montrer un joueur effectuant un coup franc avec un ballon visible.  
        **Préparez-vous pour une analyse de niveau pro !** 🎯
    """, unsafe_allow_html=True)

    status_container = st.empty()

    uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi"], key="video_uploader")

    if uploaded_file is not None:
        if validate_video_file(uploaded_file):
            try:
                status_container.write("Aperçu de la vidéo téléchargée:")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
                    tmp_input.write(uploaded_file.read())
                    input_path = tmp_input.name
                    st.video(input_path)

                output_path = tempfile.mktemp(suffix=".mp4")
                csv_path = tempfile.mktemp(suffix=".csv")

                if st.button("Analyser la vidéo", key="process_button"):
                    with st.spinner("Analyse en cours..."):
                        success = process_video(input_path, output_path, csv_path, status_container)

                    if success:
                        status_container.write("Vidéo annotée générée :")
                        st.video(output_path)

                        col1, col2 = st.columns(2)
                        with col1:
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="Télécharger la vidéo annotée",
                                    data=f,
                                    file_name="predicted_video.mp4",
                                    mime="video/mp4",
                                    key="download_video"
                                )
                        with col2:
                            with open(csv_path, "rb") as f:
                                st.download_button(
                                    label="Télécharger le fichier CSV",
                                    data=f,
                                    file_name="angles_predictions.csv",
                                    mime="text/csv",
                                    key="download_csv"
                                )
                        # Ajouter un effet de célébration
                        st.balloons()
                        st.success("Great Analysis! ⚽ Ready for the next goal? 🥅")

                    try:
                        os.unlink(input_path)
                        os.unlink(output_path)
                        os.unlink(csv_path)
                    except Exception as e:
                        logger.warning(f"Erreur lors du nettoyage des fichiers temporaires: {e}")

            except Exception as e:
                logger.error(f"Erreur lors du chargement ou traitement de la vidéo: {e}")
                st.error(f"Erreur lors du chargement ou traitement de la vidéo: {e}")
            finally:
                holistic.close()

if __name__ == "__main__":
    main()