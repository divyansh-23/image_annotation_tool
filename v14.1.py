import os
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import sqlite3
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import json
import pickle
import io

# Create a function to save the DataFrame to Excel


def save_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Select the columns you want to save
        columns_to_save = ["weld", "surface_quality",
                           "blow_out", "melt_pool_overflow", "weld_through"]
        df[columns_to_save].to_excel(writer, index=False)
    return output.getvalue()


def serialize_df(df):
    return pickle.dumps(df)


def deserialize_df(df_blob):
    return pickle.loads(df_blob)


def serialize_json(json_data):
    return json.dumps(json_data)


def deserialize_json(json_blob):
    return json.loads(json.loads(json_blob))


# SETTINGS
# Increase the max number of columns displayed
pd.set_option("display.max_columns", None)
# Increase the max width of each column
pd.set_option("display.max_colwidth", None)


def subexperiment_exists(matrix, subexperiment):
    for row in matrix:
        if row[1] == subexperiment:
            return True
    return False


def update_matrix(matrix, subexperiment, data):
    if subexperiment_exists(matrix, subexperiment):
        for i, row in enumerate(matrix):
            if row[1] == subexperiment:
                matrix[i] = data
    else:
        matrix.append(data)


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.canvas_image = {}
        self.image_index = 0
        self.table_data = []
        self.intensity_images = []
        self.display_df = False


def reset_canvas(image_index, experiment_df):
    st.session_state.image_index = image_index
    st.session_state.canvas_image = {}
    st.experimental_rerun()

    experiment_df.loc[:, ["weld", "surface_quality",
                          "blow_out", "melt_pool_overflow", "weld_through"]] = 0
    st.session_state.canvas_image.json_data = {"objects": []}
    st.success("All annotations have been reset!")


def load_intensity_image(image_path):
    image = Image.open(image_path)
    if image.mode == "RGB":
        image_np = np.array(image)
    elif image.mode == "L":
        image_np = np.array(image.convert("RGB"))
    else:
        raise ValueError("Unsupported image mode: {}".format(image.mode))
    return image_np


def get_start_end_coordinates(image):
    start_x = 0
    start_y = 0
    end_x = image.shape[0]
    end_y = image.shape[1]
    return start_x, start_y, end_x, end_y


def capture_weld_logic(intensity_image_path, experiment_df, experiment_folder, start_x, start_y, end_x, end_y, subexperiment_folder, scaling_factor_x, scaling_factor_y, rescaled_width, rescaled_height):
    if st.session_state.canvas_image.image_data is not None:
        pp = []
        if st.session_state.canvas_image.json_data is not None:
            for shape in st.session_state.canvas_image.json_data["objects"]:
                if shape["type"] == "circle" and shape["stroke"] == "#FF0000":
                    y1, x1 = shape["left"], shape["top"]
                    pp.append((x1, y1))
                    # display error if more than 2 points are selected
                    if len(pp) > 2:
                        st.error("More than 2 points selected")
                        return

        if len(pp) == 2:
            welding_start_x = pp[0][0] / scaling_factor_x
            welding_start_y = pp[0][1] / scaling_factor_y
            welding_stop_x = pp[1][0] / scaling_factor_x
            welding_stop_y = pp[1][1] / scaling_factor_y
            st.session_state['weld_df'].loc[welding_start_x:welding_stop_x+1, "weld"] = 1
            experiment_df.loc[welding_start_x:welding_stop_x+1, "weld"] = 1
            st.session_state['experiment_dfs'] = pd.concat([st.session_state['weld_df'], st.session_state['surface_quality_df'],
                                                            st.session_state['blowout_df'], st.session_state['melt_pool_overflow_df'],
                                                            st.session_state['weld_through_df']], axis=1)


def main():
    if 'display_df' not in st.session_state:
        st.session_state.display_df = False
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # Establish a connection to the database
    conn = sqlite3.connect(
        './data/image_annotation_tool_jeno_test1.db')
    cursor = conn.cursor()

    # Create a table for storing pandas dataframes for each image
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataframes (
            image_id TEXT PRIMARY KEY,
            dataframe BLOB
        )
    """)

    # Create a table for storing the JSON data of the canvas for each image
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS json_data (
            image_id TEXT PRIMARY KEY,
            json_data BLOB
        )
    """)

    create_quality_table_sql = """
    CREATE TABLE IF NOT EXISTS overall_quality (
        image_id TEXT PRIMARY KEY,
        quality INTEGER
    )
    """
    cursor.execute(create_quality_table_sql)
    conn.commit()

    st.title("Triangulation Pre-Trials Application")

    # Hard-code the main folder path
    main_folder_path = "./Triangulation_PreTrials"
    # *** CHANGE THE MAIN FOLDER PATH: https://www.kaggle.com/code/sarahmacleans/detection-of-weld-defects/notebook

s    if 'intensity_images' not in st.session_state:
        # Find all intensity images in the main folder path
        intensity_images = []
        for root, dirs, files in os.walk(main_folder_path):
            for file in files:
                if "INTENSITY" in file:
                    intensity_images.append(os.path.join(root, file))

        intensity_images = sorted(intensity_images)

        st.session_state['intensity_images'] = intensity_images
    else:
        intensity_images = st.session_state['intensity_images']

    # Check if there are intensity images
    if not intensity_images:
        st.error("No intensity images found.")
        return

    # Check if the image index is within the intensity images range
    image_index = st.session_state.get("image_index", 0)

    if image_index >= len(intensity_images):
        st.info("All images processed.")
        return

    # Load the intensity image
    intensity_image_path = intensity_images[image_index]
    intensity_image = load_intensity_image(intensity_image_path)

    # Find the experiment and subexperiment values
    relative_path = os.path.relpath(intensity_image_path, main_folder_path)
    experiment_folder, subexperiment_folder = relative_path.split(os.sep)[:2]

    subexperiment_name = subexperiment_folder.split(".")[0]

    # Extract only the required subexperiment value
    subexperiment_value = subexperiment_folder.split("_")[1]

    st.header(
        f"Experiment: {experiment_folder} | Sub-Experiment: {subexperiment_value}")
    st.markdown("""
                <i class="fas fa-info-circle" style="color:blue" onclick="alert('Here is some info!')"></i>
                """,
                unsafe_allow_html=True)

    with st.expander("Step 1: Capture welding 'start' and 'end' points:"):
        st.write("Mark the two points on the image and click on 'Capture welding start and end points' button. The points will be marked in RED color")

    with st.expander("Step 2: Capture 'surface quality':"):
        st.write("Select a value from the sidemenu on the left and draw the vertical lines on the image along the weld region to annotate the surface quality of a region. The Surface Quality will be marked in the color selected from the sidemenu.")

    with st.expander("Step 3: Capture 'blowout':"):
        st.write("Select the 'Blowout' option from the sidemenu on the left and draw a circle around the blowout in the welding region in the image. The Blowout will be marked in CYAN color.")

    with st.expander("Step 4: Capture 'Outside':"):
        st.write("Select the 'Melt Pool Overflow' option from the sidemenu on the left and draw a circle around the melt pool overflow in the welding region in the image. The Melt Poll Overflow will be marked in MAGENTA color ")

    welding_start_x = 0
    welding_start_y = 0
    welding_stop_x = 0
    welding_stop_y = 0
    start_x, start_y, end_x, end_y = get_start_end_coordinates(
        intensity_image)

    # Try to retrieve the dataframe from the database
    cursor.execute(
        "SELECT dataframe FROM dataframes WHERE image_id = ?", (subexperiment_name,))
    df_row = cursor.fetchone()
    if df_row is not None:
        df_blob = df_row[0]
        # Use the deserialize_df function here
        experiment_df = deserialize_df(df_blob)
    else:
        experiment_df = pd.DataFrame(
            columns=["weld", "surface_quality", "blow_out", "melt_pool_overflow", "weld_through"], index=range(0, end_x))
        experiment_df = experiment_df.applymap(lambda x: 0)

    # Initialize DataFrames for each type of annotation
    st.session_state['weld_df'] = pd.DataFrame(
        columns=["weld"], index=range(0, end_x)).applymap(lambda x: 0)
    st.session_state['surface_quality_df'] = pd.DataFrame(
        columns=["surface_quality"], index=range(0, end_x)).applymap(lambda x: 0)
    st.session_state['blowout_df'] = pd.DataFrame(
        columns=["blow_out"], index=range(0, end_x)).applymap(lambda x: 0)
    st.session_state['melt_pool_overflow_df'] = pd.DataFrame(
        columns=["melt_pool_overflow"], index=range(0, end_x)).applymap(lambda x: 0)
    st.session_state['weld_through_df'] = pd.DataFrame(
        columns=["weld_through"], index=range(0, end_x)).applymap(lambda x: 0)

    # Add the "blowout" to the DataFrame
    experiment_df["blow_out"] = 0
    # Add the "weld_through" to the DataFrame
    experiment_df["weld_through"] = 0
    # Add the "melt_pool_overflow" to the DataFrame
    experiment_df["melt_pool_overflow"] = 0

    scaling_factor_y = 344 / end_y
    scaling_factor_x = 709.44 / end_x
    rescaled_height = end_x * scaling_factor_x
    rescaled_width = end_y * scaling_factor_y

    if "canvas_image" not in st.session_state:
        st.session_state.canvas_image = {}

    # Check if the button has been pressed
    if 'is_captured' not in st.session_state:
        st.session_state['is_captured'] = False

    annotation_mode = st.sidebar.selectbox(
        "Annotation tool:", ("Weld", "Surface Quality", "Blowout",
                             "Melt Pool Overflow", "Delete Selected Annotation", "Weld Through")
    )

    def get_drawing_mode():
        if annotation_mode == "Weld":
            return "point"
        elif annotation_mode == "Surface Quality" or annotation_mode == "Weld Through":
            return "line"
        elif annotation_mode == "Delete Selected Annotation":
            return "transform"
        elif annotation_mode == "Blowout" or annotation_mode == "Melt Pool Overflow":
            return "circle"

    if annotation_mode == "Surface Quality":
        quality_color = st.sidebar.radio("Choose a Value to annotate Surface Quality for a region", [
                                         "5 - dark green", "4 - green", "3 - yellow", "2 - orange", "1 - pink"])

    def get_SR_quality_color():
        if quality_color == "5 - dark green":
            return "#006400"
        elif quality_color == "4 - green":
            return "#90EE90"
        elif quality_color == "3 - yellow":
            return "#FFFF00"
        elif quality_color == "2 - orange":
            return "#FFA500"
        elif quality_color == "1 - pink":
            return "#FFC0CB"

    def get_SA_value_from_color(color):
        if color == "#006400":
            # blue color
            return 5
        elif color == "#90EE90":
            # green color
            return 4
        elif color == "#FFFF00":
            # orange color
            return 3
        elif color == "#FFA500":
            # yellow color
            return 2
        elif color == "#FFC0CB":
            # red color
            return 1

    def get_stroke_color(annotation_mode):
        if annotation_mode == "Surface Quality":
            return get_SR_quality_color()
        elif annotation_mode == "Blowout":
            # annotating blowout with cyan color
            return "#00FFFF"
        elif annotation_mode == "Melt Pool Overflow":
            # annotating melt pool overflow with magenta color
            return "#FF00FF"
        elif annotation_mode == "Weld":
            # annotating weld with red color
            return "#FF0000"
        elif annotation_mode == "Weld Through":
            # annotating new annotation with white color
            return "#FFFFFF"
        else:
            return "#FF0000"

    def update_surface_quality_color(annotation):
        if annotation["type"] == "line":
            old_color = annotation["stroke"]
            if old_color == "#0000FF":
                annotation["stroke"] = "#006400"
            elif old_color == "#00FF00":
                annotation["stroke"] = "#90EE90"
            if old_color == "#FFA500":
                annotation["stroke"] = "#FFFF00"
            if old_color == "#FF0000":
                annotation["stroke"] = "#FFA500"
            if old_color == "#FF0000":
                annotation["stroke"] = "#FFC0CB"

    def show_dataframe():
        st.subheader("Single Dataframe")
        st.dataframe(experiment_df)
        json_data = json.dumps(st.session_state.canvas_image.json_data)
        save_data(subexperiment_name, experiment_df, json_data, conn)
        # load_data(subexperiment_name, conn)

    stroke_color_used = get_stroke_color(annotation_mode)

    # display the quality image for the user
    quality_image = Image.open(
        './Triangulation_PreTrials/image_quality.png')
    st.image(quality_image, caption='Quality reference image')

    # After loading the image and before displaying the st_canvas
    overall_quality = get_overall_quality_from_db(subexperiment_name, conn)
    if overall_quality is not None:
        st.write(f"Overall Quality: {overall_quality}")
    else:
        st.write("No overall quality score assigned yet.")

    # Give an option to assign a new score
    new_quality = st.number_input(
        "Assign a new overall quality score (1-5)", min_value=1, max_value=5)
    if st.button("Save new score"):
        add_overall_quality_to_db(subexperiment_name, new_quality, conn)
        st.write("New overall quality score saved.")

    cursor.execute(
        "SELECT json_data FROM json_data WHERE image_id = ?", (subexperiment_name,))
    json_row = cursor.fetchone()
    stored_json_data = deserialize_json(
        json_row[0]) if json_row is not None else {}

    st.session_state.canvas_image = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=5,
        stroke_color=stroke_color_used,
        background_image=Image.open(intensity_image_path),
        width=rescaled_width,
        height=rescaled_height,
        drawing_mode=get_drawing_mode(),
        point_display_radius=2 if get_drawing_mode() == 'point' else 0,
        initial_drawing=stored_json_data,
        display_toolbar=True,
        key=f"canvas_{subexperiment_folder}"
    )

    capture_weld_logic(intensity_image_path, experiment_df, experiment_folder, start_x, start_y, end_x,
                       end_y, subexperiment_folder, scaling_factor_x, scaling_factor_y, rescaled_width, rescaled_height)

    if st.session_state.canvas_image.image_data is not None:
        if st.session_state.canvas_image.json_data is not None:
            for shape in st.session_state.canvas_image.json_data["objects"]:
                if shape["type"] == "line" and annotation_mode == "Surface Quality":
                    update_surface_quality_color(shape)
                    y1 = (shape['top'] + shape['y1']) / scaling_factor_x
                    y2 = (shape['top'] + shape['y2']) / scaling_factor_x
                    value = get_SA_value_from_color(shape["stroke"])
                    st.session_state['surface_quality_df'].loc[y1:y2 +
                                                               1, "surface_quality"] = value
                    experiment_df.loc[y1:y2+1, "surface_quality"] = value
                    st.session_state['experiment_dfs'] = pd.concat([st.session_state['weld_df'], st.session_state['surface_quality_df'],
                                                                    st.session_state['blowout_df'], st.session_state['melt_pool_overflow_df'],
                                                                    st.session_state['weld_through_df']], axis=1)

    if st.session_state.canvas_image.image_data is not None:
        if st.session_state.canvas_image.json_data is not None:
            for shape in st.session_state.canvas_image.json_data["objects"]:
                if shape["type"] == "line" and annotation_mode == "Weld Through":
                    update_surface_quality_color(shape)
                    y1 = (shape['top'] + shape['y1']) / scaling_factor_x
                    y2 = (shape['top'] + shape['y2']) / scaling_factor_x
                    st.session_state['weld_through_df'].loc[y1:y2 +
                                                            1, "weld_through"] = 1
                    experiment_df.loc[y1:y2+1, "weld_through"] = 1
                    st.session_state['experiment_dfs'] = pd.concat([st.session_state['weld_df'], st.session_state['surface_quality_df'],
                                                                    st.session_state['blowout_df'], st.session_state['melt_pool_overflow_df'],
                                                                    st.session_state['weld_through_df']], axis=1)

    if st.session_state.canvas_image.image_data is not None:
        if st.session_state.canvas_image.json_data is not None:
            for shape in st.session_state.canvas_image.json_data["objects"]:
                if shape["type"] == "circle" and shape["stroke"] == "#00FFFF":
                    y1 = (shape['top']) / scaling_factor_x
                    y2 = (shape['top'] + shape['height']) / scaling_factor_x
                    st.session_state['blowout_df'].loc[y1:y2 +
                                                       1, "blow_out"] = 1
                    experiment_df.loc[y1:y2+1, "blow_out"] = 1
                    st.session_state['experiment_dfs'] = pd.concat([st.session_state['weld_df'], st.session_state['surface_quality_df'],
                                                                    st.session_state['blowout_df'], st.session_state['melt_pool_overflow_df'],
                                                                    st.session_state['weld_through_df']], axis=1)

    if st.session_state.canvas_image.image_data is not None:
        if st.session_state.canvas_image.json_data is not None:
            for shape in st.session_state.canvas_image.json_data["objects"]:
                if shape["type"] == "circle" and shape["stroke"] == "#FF00FF":
                    y1 = (shape['top'] - shape['radius']) / \
                        scaling_factor_x
                    y2 = (shape['top'] + shape['radius']) / \
                        scaling_factor_x
                    st.session_state['melt_pool_overflow_df'].loc[y1:y2 +
                                                                  1, "melt_pool_overflow"] = 1
                    experiment_df.loc[y1:y2+1, "melt_pool_overflow"] = 1
                    st.session_state['experiment_dfs'] = pd.concat([st.session_state['weld_df'], st.session_state['surface_quality_df'],
                                                                    st.session_state['blowout_df'], st.session_state['melt_pool_overflow_df'],
                                                                    st.session_state['weld_through_df']], axis=1)

    # Add a button to download as Excel
    if st.button("Download Dataframe as Excel"):
        excel_data = save_excel(experiment_df)
        st.download_button(
            label="Download Excel",
            data=excel_data,
            key="download-excel",
            file_name="experiment_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    col1, col2, col3 = st.columns(4)

    with col1:
        if st.button("Previous Image"):
            # Decrement the image index
            image_index = st.session_state.get("image_index", 1)
            image_index -= 1
            if image_index >= 0:
                st.session_state.image_index = image_index
                # Reset the canvas and input values
                json_data = json.dumps(st.session_state.canvas_image.json_data)
                save_data(subexperiment_name, experiment_df, json_data, conn)
                reset_canvas(st.session_state.image_index, experiment_df)

    with col2:
        if st.button("Next Image"):
            image_index = st.session_state.get("image_index", 0)
            image_index += 1
            st.session_state.image_index = image_index

            # #  Reset the canvas and input values
            json_data = json.dumps(st.session_state.canvas_image.json_data)
            save_data(subexperiment_name, experiment_df, json_data, conn)
            reset_canvas(st.session_state.image_index, experiment_df)

            # cursor.close()

    with col3:
        if st.button("Clear Complete Image Canvas"):
            cursor = conn.cursor()

            # Execute SQL command to delete the record for the given image_id from dataframes table
            cursor.execute("DELETE FROM dataframes WHERE image_id = ?",
                           (subexperiment_name,))

            # Execute SQL command to delete the record for the given image_id from json_data table
            cursor.execute("DELETE FROM json_data WHERE image_id = ?",
                           (subexperiment_name,))

            # Commit the changes and close the connection
            conn.commit()
            st.experimental_rerun()

    with col4:
        if False and st.button("Display DF"):
            # This is just to handle the button click event
            st.session_state.display_df = True

    # Now, outside of the columns:
    if st.session_state.get('display_df', False):
        st.subheader("Annotated Dataframe")
        st.dataframe(experiment_df)
        json_data = json.dumps(st.session_state.canvas_image.json_data)
        save_data(subexperiment_name, experiment_df, json_data, conn)

    # Input field for user to enter the index number of the sub-experiment
    subexp_index = st.number_input(
        f"Enter sub-experiment index between 0 and {len(intensity_images) - 1}", value=0, step=1)

    # Button to move to the entered sub-experiment index
    if st.button("Go to sub-experiment"):
        # If the index is within the range of total sub-experiments, change the current image_index to the entered index
        if 0 <= subexp_index < len(intensity_images):
            st.session_state.image_index = subexp_index

            # Reset the canvas and input values based on the new sub-experiment
            json_data = json.dumps(st.session_state.canvas_image.json_data)
            save_data(subexperiment_name, experiment_df, json_data, conn)
            reset_canvas(st.session_state.image_index, experiment_df)
        else:
            st.error("Invalid sub-experiment index")


def save_data(image_id, df, json_data, conn):
    # Serialize the dataframe and JSON data
    df_blob = serialize_df(df)
    json_blob = serialize_json(json_data)

    with sqlite3.connect('./data/image_annotation_tool_jeno_test1.db') as conn:
        cursor = conn.cursor()

        # Insert the dataframe into the 'dataframes' table
        cursor.execute(
            "INSERT OR REPLACE INTO dataframes VALUES (?, ?)", (image_id, df_blob))

        # Insert the JSON data into the 'json_data' table
        cursor.execute(
            "INSERT OR REPLACE INTO json_data VALUES (?, ?)", (image_id, json_blob))


def load_data(image_id, conn):
    with sqlite3.connect('./data/image_annotation_tool_jeno_test1.db') as conn:
        # Create a cursor
        cursor = conn.cursor()

        # Retrieve the dataframe from the 'dataframes' table
        cursor.execute(
            "SELECT * FROM dataframes WHERE image_id=?", (image_id,))
        df_blob = cursor.fetchone()[1]
        df = deserialize_df(df_blob)

        # Retrieve the JSON data from the 'json_data' table
        cursor.execute("SELECT * FROM json_data WHERE image_id=?", (image_id,))
        json_blob = cursor.fetchone()[1]
        json_data = deserialize_json(json_blob)

        # Close the cursor
        cursor.close()

    return df, json_data


def add_overall_quality_to_db(image_id, quality, conn):
    with sqlite3.connect('./data/image_annotation_tool_jeno_test1.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO overall_quality VALUES (?, ?)", (image_id, quality))


def get_overall_quality_from_db(image_id, conn):
    with sqlite3.connect('./data/image_annotation_tool_jeno_test1.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT quality FROM overall_quality WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    main()
