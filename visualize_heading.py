import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import folium
    import math
    #from folium.plugins import BeautyTimer

    def create_heading_map(latitude, longitude, heading_degrees, output_file='heading_map.html'):
        """
        Crée une carte avec un point et une flèche indiquant le heading

        Args:
            latitude: Latitude du point
            longitude: Longitude du point
            heading_degrees: Angle de heading en degrés (0-360)
            output_file: Nom du fichier HTML de sortie
        """
        # Création de la carte centrée sur le point
        m = folium.Map(location=[latitude, longitude], zoom_start=15)
    
        # Ajout du point
        folium.Marker(
            location=[latitude, longitude],
            popup=f"Point (Lat: {latitude}, Lon: {longitude})",
            icon=folium.Icon(color='blue', icon='info-sign'),
        ).add_to(m)

        # Calcul de la direction de la flèche
        heading_rad = math.radians(heading_degrees)
        arrow_length = 0.005  # Longueur de la flèche en degrés (environ 500m à l'équateur)

        # Coordonnées de la pointe de la flèche
        end_lat = latitude + math.cos(heading_rad) * arrow_length
        end_lon = longitude + math.sin(heading_rad) * arrow_length

        # Ajout de la flèche
        folium.PolyLine(
            locations=[[latitude, longitude], [end_lat, end_lon]],
            color='red',
            weight=5,
            opacity=0.8
        ).add_to(m)

        # Ajout d'une flèche à la pointe
        folium.RegularPolygonMarker(
            location=[end_lat, end_lon],
            number_of_sides=3,
            radius=0.001,
            rotation=heading_degrees,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)

        # Ajout d'une légende
        legend_html = f"""
        <div style="position: fixed;
                     bottom: 50px; left: 50px; width: 200px; height: 80px;
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white; padding: 10px;">
        <b>Heading:</b> {heading_degrees}°<br>
        <b>Point:</b> {latitude:.6f}, {longitude:.6f}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Sauvegarde de la carte
        m.save(output_file)
        print(f"Carte sauvegardée dans {output_file}")

    return (create_heading_map,)


@app.cell
def _(create_heading_map):
    # Exemple d'utilisation



    # Coordonnées d'un point (exemple: Tour Eiffel)
    latitude = 22.892861
    longitude = -109.840437

    # Heading (cap) en degrés (exemple: 45° par rapport au nord)
    heading_degrees = -50.9

    # DJI_20260116154954_0001_V.MP4 - Frame 7352
    latitude, longitude, heading_degrees = (22.891129, -109.842292, -113.3)
    # DJI_20260116154954_0001_V.MP4 - Frame 7506
    latitude, longitude, heading_degrees = (22.891129, -109.842292, -126.4)
    # Création de la carte
    create_heading_map(latitude, longitude, heading_degrees)
    return


if __name__ == "__main__":
    app.run()
