import os
import sys
import shutil
import subprocess
import getpass
from pathlib import Path

def get_credentials():
    """Intenta obtener credenciales de Secrets (Kaggle/Colab) o pide Input."""
    user, password = None, None
    
    # 1. Intentar KAGGLE SECRETS
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        user = secrets.get_secret("DIFFLOCKS_USERNAME")
        password = secrets.get_secret("DIFFLOCKS_PASSWORD")
        if user and password:
            print("   🔑 Credenciales encontradas en Kaggle Secrets.")
            return user, password
    except:
        pass

    # 2. Intentar COLAB SECRETS
    try:
        from google.colab import userdata
        user = userdata.get("DIFFLOCKS_USERNAME")
        password = userdata.get("DIFFLOCKS_PASSWORD")
        if user and password:
            print("   🔑 Credenciales encontradas en Google Colab Secrets.")
            return user, password
    except:
        pass

    # 3. FALLBACK: INPUT MANUAL (Para Local)
    print("\n⚠️ No se detectaron Secrets configurados.")
    print("   Por favor ingresa tu cuenta de https://difflocks.is.tue.mpg.de/")
    user = input("   👤 Username (DiffLocks): ").strip()
    password = getpass.getpass("   🔑 Password (DiffLocks): ").strip()
    return user, password

def run():
    print("⬇️ [DiffLocks] Gestor de Checkpoints")
    base_dir = Path(os.getcwd())
    ckpt_dst = base_dir / "checkpoints"
    
    # --- FASE 1: BUSCAR CACHÉ (Kaggle Input o Local) ---
    print("   🔎 Buscando checkpoints existentes...")
    
    # Lista de lugares donde buscar
    search_paths = [
        Path("/kaggle/input"),          # Kaggle Datasets
        Path("/content/drive"),         # Colab Drive
        base_dir.parent,                # Carpeta superior
        base_dir                        # Carpeta actual
    ]
    
    ckpt_src = None
    for path in search_paths:
        if path.exists():
            # Buscamos el archivo clave 'scalp_v9_40k...'
            for f in path.rglob("scalp_v9_40k_*.pth"):
                ckpt_src = f.parent.parent
                break
        if ckpt_src: break
        
    if ckpt_src:
        # Enlazar y salir
        if ckpt_dst.exists() or ckpt_dst.is_symlink():
            if ckpt_dst.is_symlink(): os.unlink(ckpt_dst)
            else: shutil.rmtree(ckpt_dst)
            
        # En Windows usamos copytree (symlinks requieren admin), en Linux symlink
        if os.name == 'nt':
            shutil.copytree(ckpt_src, ckpt_dst)
        else:
            os.symlink(ckpt_src, ckpt_dst)
            
        print(f"✅ ¡ÉXITO! Usando caché encontrado en: {ckpt_src}")
        return

    # --- FASE 2: DESCARGA (Si no hay caché) ---
    print("   ⚠️ No se encontró caché local. Iniciando descarga...")
    
    user, password = get_credentials()
    
    if not user or not password:
        print("❌ Error: Se requieren credenciales para descargar.")
        sys.exit(1)
        
    # URL Encoding
    import urllib.parse
    user_enc = urllib.parse.quote(user)
    pass_enc = urllib.parse.quote(password)
    
    url = "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=difflocks_checkpoints.zip"
    zip_name = "difflocks_checkpoints.zip"
    
    print(f"   ⏳ Descargando checkpoints...")
    
    # Comando wget compatible
    cmd = [
        "wget", 
        "--post-data", f"username={user_enc}&password={pass_enc}",
        url, 
        "-O", zip_name,
        "--no-check-certificate", 
        "--continue",
        "-q", "--show-progress"
    ]
    
    try:
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise Exception("Código de error en wget")
            
        print("   📦 Descomprimiendo...")
        subprocess.run(["unzip", "-q", "-o", zip_name])
        
        if os.path.exists(zip_name):
            os.remove(zip_name)
            
        if ckpt_dst.exists():
            print("✅ ¡ÉXITO! Checkpoints descargados e instalados.")
        else:
            print("❌ Error: La descompresión falló.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error en la descarga: {e}")
        print("   Verifica tu usuario y contraseña.")
        sys.exit(1)

if __name__ == "__main__":
    run()
