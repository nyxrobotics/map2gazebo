#!/usr/bin/env python

import os
from math import atan2
from math import pi

import cv2
import numpy as np
import rospy
import trimesh
from nav_msgs.msg import OccupancyGrid


class MapConverter(object):
    def __init__(self, map_topic, threshold=1, height=2.0, use_contours=False):
        self.test_map_pub = rospy.Publisher(
            "test_map", OccupancyGrid, latch=True, queue_size=1)
        self.threshold = threshold
        self.height = height
        self.use_contours = use_contours
        map_msg = rospy.wait_for_message(map_topic, OccupancyGrid)
        self.create_map(map_msg)

    def create_map(self, map_msg):
        mesh_type = rospy.get_param("~mesh_type", "stl")
        export_dir = rospy.get_param("~export_dir")
        file_name = rospy.get_param("~file_name", "map")
        if mesh_type not in ["stl", "dae"]:
            rospy.logerr("Invalid mesh type: {}".format(mesh_type))
            return
        if not os.path.exists(export_dir):
            rospy.logerr(f"Export directory {export_dir} does not exist")
            return
        rospy.loginfo("Received map")
        map_dims = (map_msg.info.height, map_msg.info.width)
        map_array = np.array(map_msg.data).reshape(map_dims)

        map_array[map_array < 0] = 0
        if self.use_contours:
            meshes = self.image_contours_to_mesh(map_array, map_msg.info)
        else:
            meshes = self.image_pixels_to_mesh(map_array, map_msg.info)

        mesh = trimesh.util.concatenate(meshes)

        if mesh_type == "stl":
            with open(export_dir + "/" + file_name + ".stl", 'wb') as f:
                mesh.export(f, "stl")
            rospy.loginfo("Exported STL: " + export_dir + "/" + file_name + ".stl")
        elif mesh_type == "dae":
            with open(export_dir + "/" + file_name + ".dae", 'w') as f:
                f.write(trimesh.exchange.dae.export_collada(mesh))
            rospy.loginfo("Exported DAE.")

    def publish_test_map(self, points, metadata, map_header):
        test_map = np.zeros((metadata.height, metadata.width))
        for x, y in points:
            test_map[y, x] = 100
        test_map_msg = OccupancyGrid()
        test_map_msg.header = map_header
        test_map_msg.header.stamp = rospy.Time.now()
        test_map_msg.info = metadata
        test_map_msg.data = list(np.ravel(test_map))
        self.test_map_pub.publish(test_map_msg)

    def get_occupied_regions(self, map_array):
        map_array = map_array.astype(np.uint8)
        _, thresh_map = cv2.threshold(map_array, self.threshold, 100, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]
        corner_idxs = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
        return [contours[i] for i in corner_idxs]

    def image_contours_to_mesh(self, image, metadata):
        height = np.array([0, 0, self.height])
        meshes = []
        contours = self.get_occupied_regions(image)
        faces_set = set()

        for contour in contours:
            vertices_dict = {}
            num_points = len(contour)
            for i, point in enumerate(contour):
                x, y = point[0]
                v0 = coords_to_loc((x, y), metadata)
                v1 = coords_to_loc((x, y + 1), metadata)
                v2 = coords_to_loc((x + 1, y), metadata)
                v3 = coords_to_loc((x + 1, y + 1), metadata)

                # Initialize wall flags
                set_wall_x_plus = True
                set_wall_x_minus = True
                set_wall_y_plus = True
                set_wall_y_minus = True

                # Check next point
                if i < num_points - 1:
                    next_point = contour[i + 1][0]
                    next_x, next_y = next_point
                    vector_x = next_y - y
                    vector_y = -(next_x - x)
                    direction = atan2(vector_y, vector_x)
                    if -pi / 4 <= direction < pi / 4:
                        set_wall_y_plus = False
                    elif pi / 4 <= direction < 3 * pi / 4:
                        set_wall_x_plus = False
                    elif -3 * pi / 4 <= direction < -pi / 4:
                        set_wall_x_minus = False
                    else:
                        set_wall_y_minus = False

                # Check previous point
                if i > 0:
                    prev_point = contour[i - 1][0]
                    prev_x, prev_y = prev_point
                    vector_x = prev_y - y
                    vector_y = -(prev_x - x)
                    direction = atan2(vector_y, vector_x)
                    if -pi / 4 <= direction < pi / 4:
                        set_wall_y_plus = False
                    elif pi / 4 <= direction < 3 * pi / 4:
                        set_wall_x_plus = False
                    elif -3 * pi / 4 <= direction < -pi / 4:
                        set_wall_x_minus = False
                    else:
                        set_wall_y_minus = False

                # Add faces based on flags
                if set_wall_x_plus:
                    self.add_face(v2, v3, v2 + height, v3 + height, np.array([0, 0, 0]), faces_set)
                if set_wall_x_minus:
                    self.add_face(v0, v1, v0 + height, v1 + height, np.array([0, 0, 0]), faces_set)
                if set_wall_y_plus:
                    self.add_face(v1, v3, v1 + height, v3 + height, np.array([0, 0, 0]), faces_set, reverse=True)
                if set_wall_y_minus:
                    self.add_face(v0, v2, v0 + height, v2 + height, np.array([0, 0, 0]), faces_set, reverse=True)

                # Add roof and floor faces
                faces_set.add((tuple(v0 + height), tuple(v2 + height), tuple(v1 + height)))
                faces_set.add((tuple(v2 + height), tuple(v3 + height), tuple(v1 + height)))
                faces_set.add((tuple(v0), tuple(v1), tuple(v2)))
                faces_set.add((tuple(v2), tuple(v1), tuple(v3)))

            vertices = set()
            faces = []
            for face in faces_set:
                face_indices = []
                for vertex in face:
                    if vertex not in vertices_dict:
                        vertices_dict[vertex] = len(vertices_dict)
                    face_indices.append(vertices_dict[vertex])
                faces.append(face_indices)

            vertices = np.array([list(vertex) for vertex in vertices_dict])
            faces = np.array(faces)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.remove_duplicate_faces()
            meshes.append(mesh)

        return meshes

    def image_pixels_to_mesh(self, image, metadata):
        height = np.array([0, 0, self.height])
        vertices_dict = {}
        faces_set = set()

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y, x] < self.threshold:
                    continue

                v0 = coords_to_loc((x, y), metadata)
                v1 = coords_to_loc((x, y + 1), metadata)
                v2 = coords_to_loc((x + 1, y), metadata)
                v3 = coords_to_loc((x + 1, y + 1), metadata)

                # Initialize wall flags
                set_wall_x_plus = y == 0 or image[y - 1, x] < self.threshold
                set_wall_x_minus = y == image.shape[0] - 1 or image[y + 1, x] < self.threshold
                set_wall_y_plus = x == image.shape[1] - 1 or image[y, x + 1] < self.threshold
                set_wall_y_minus = x == 0 or image[y, x - 1] < self.threshold

                # Add faces based on flags
                if set_wall_x_plus:
                    self.add_face(v0, v2, v0 + height, v2 + height, np.array([0, 0, 0]), faces_set)
                if set_wall_y_plus:
                    self.add_face(v2, v3, v2 + height, v3 + height, np.array([0, 0, 0]), faces_set)
                if set_wall_x_minus:
                    self.add_face(v1, v3, v1 + height, v3 + height, np.array([0, 0, 0]), faces_set, reverse=True)
                if set_wall_y_minus:
                    self.add_face(v0, v1, v0 + height, v1 + height, np.array([0, 0, 0]), faces_set, reverse=True)

                # Roof face
                faces_set.add((tuple(v0 + height), tuple(v2 + height), tuple(v1 + height)))
                faces_set.add((tuple(v2 + height), tuple(v3 + height), tuple(v1 + height)))

                # Floor face
                faces_set.add((tuple(v0), tuple(v1), tuple(v2)))
                faces_set.add((tuple(v2), tuple(v1), tuple(v3)))

        vertices = set()
        faces = []
        for face in faces_set:
            face_indices = []
            for vertex in face:
                if vertex not in vertices_dict:
                    vertices_dict[vertex] = len(vertices_dict)
                face_indices.append(vertices_dict[vertex])
            faces.append(face_indices)

        vertices = np.array([list(vertex) for vertex in vertices_dict])
        faces = np.array(faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.remove_duplicate_faces()

        return mesh

    def add_face(self, v0, v1, v2, v3, height_offset, faces_set, reverse=False):
        v0 = tuple(v0 + height_offset)
        v1 = tuple(v1 + height_offset)
        v2 = tuple(v2 + height_offset)
        v3 = tuple(v3 + height_offset)
        if reverse:
            faces_set.add((v3, v1, v2))
            faces_set.add((v2, v1, v0))
        else:
            faces_set.add((v0, v1, v2))
            faces_set.add((v2, v1, v3))


def coords_to_loc(coords, metadata):
    x, y = coords
    loc_x = x * metadata.resolution + metadata.origin.position.x
    loc_y = y * metadata.resolution + metadata.origin.position.y
    return np.array([loc_x, loc_y, 0.0])


def main():
    rospy.init_node("map2gazebo")
    map_topic = rospy.get_param("~map_topic", "map")
    occupied_thresh = rospy.get_param("~occupied_thresh", 1)
    box_height = rospy.get_param("~box_height", 2.0)
    use_contours = rospy.get_param("~use_contours", True)
    rospy.loginfo("map2gazebo running")
    MapConverter(map_topic, threshold=occupied_thresh, height=box_height, use_contours=use_contours)


if __name__ == "__main__":
    main()
