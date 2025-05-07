# Zum benutzen des Skripts folgendes is das terminal eingeben:
#/usr/bin/python3 <<"Pfad der Funktion">> <<"Pfad des Bildes">>



from PIL import Image
import sys


def display_image_properties(image_path):
    try:
        img = Image.open(image_path)
        print(f"Bildgröße: {img.size} (Breite x Höhe)")         # Bildgröße in Pixel
        print(f"Farbmodus: {img.mode}")                         # Farbmodus (z.B. RGB, L, CMYK)
        print(f"Format: {img.format}")                          # Bildformat (z.B. TIFF, JPEG)

        # Bit-Tiefe pro Kanal (basierend auf Farbmodus)
        if img.mode == "1":
            bits_per_channel = 1
        elif img.mode in ("L", "P"):  # Graustufen (L) oder Palette (P in Gifs)
            bits_per_channel = 8
        elif img.mode in ("RGB", "RGBA", "CMYK"):
            bits_per_channel = 8
        elif img.mode == "I":  # Ganzzahlige Pixel (Integer)
            bits_per_channel = 32
        elif img.mode == "F":  # Fließkommazahlen (Float)
            bits_per_channel = 32
        else:
            bits_per_channel = 0  # Unbekannt

        # Anzahl der Kanäle ermitteln
        if img.mode in ("L", "P", "1", "I", "F"):  # Graustufen, Palette, 1-Bit, Integer, Float
            num_channels = 1  # Graustufen oder Single-Channel
        elif img.mode == "RGB":
            num_channels = 3
        elif img.mode == "RGBA":
            num_channels = 4
        elif img.mode == "CMYK":
            num_channels = 4
        else:
            num_channels = 0  # Unbekannt

        # Gesamt-Bits pro Pixel berechnen
        total_bits_per_pixel = bits_per_channel * num_channels
        print(f"Bit-Tiefe pro Kanal: {bits_per_channel}-Bit")
        print(f"Gesamt-Bits pro Pixel: {total_bits_per_pixel}-Bit")

        # Metadaten
        print("Metadaten:")
        for key, value in img.info.items():
            print(f"  {key}: {value}")

    except FileNotFoundError:
        print("Fehler: Datei nicht gefunden.")
    except OSError as e:
        print(f"Fehler beim Öffnen: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        print("Bitte den Pfad des Bildes angeben.")
        sys.exit(1)

    display_image_properties(image_path)