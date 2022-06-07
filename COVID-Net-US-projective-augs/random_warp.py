import copy
import numpy as np
from numpy.core.numeric import base_repr
import pandas as pd
from skimage import transform
import torchvision.transforms.functional as TF


class WarpConvexUS:
    def __init__(
        self,
        points_label_csv,
        max_slope=2.5,
        min_slope=0.75,
        random_slope_scale=0.15,
        min_distance_on_top=100,
    ):
        self.max_slope = max_slope
        self.min_slope = min_slope
        self.random_slope_scale = random_slope_scale
        self.min_distance_on_top = min_distance_on_top
        self.points_info = self.get_point_label_info(points_label_csv)

    def get_point_label_info(self, points_label_csv):

        if type(points_label_csv) == list:
            df = pd.concat([pd.read_csv(file, index_col=0) for file in points_label_csv])
        else:
            df = pd.read_csv(points_label_csv, index_col=0)
        points = df.apply(
            lambda x: np.array(
                [
                    [x["x1_left"], x["y1_left"]],
                    [x["x2_left"], x["y2_left"]],
                    [x["x1_right"], x["y1_right"]],
                    [x["x2_right"], x["y2_right"]],
                ]
            ),
            axis=1,
        )
        points_info = points.apply(lambda x: self.get_points_info(x))

        return {
            str(int(vid_id)): point_info
            for vid_id, point_info in zip(df.vid_id, points_info)
        }

    def get_center_point(self, corner_points, b_left=None, slope=None):
        center_x = np.mean([corner_points[0, 0], corner_points[2, 0]])
        if slope is None:
            slope = self.get_slope(corner_points)
        if b_left is None:
            b_left, b_right = self.get_b_left_right(slope, corner_points)

        center_y = -1 * slope * center_x + b_left

        return center_x, center_y

    def get_b_left_right(self, slope, corner_points):
        b_left = corner_points[1, 1] + (slope * corner_points[1, 0])
        b_right = corner_points[3, 1] - (slope * corner_points[3, 0])

        return b_left, b_right

    def get_slope(self, corner_points):
        return abs(
            (corner_points[1, 1] - corner_points[0, 1])
            / (corner_points[1, 0] - corner_points[0, 0])
        )

    def get_points_info(self, corner_points):
        return {
            "corner_points": corner_points,
            "perform_transform": False,
            "slope": 0,
            "min_slope": 0,
        }

    def get_new_corner_points(self, new_slope, orig_corner_points):
        b_left, b_right = self.get_b_left_right(new_slope, orig_corner_points)

        new_corner_points = copy.deepcopy(orig_corner_points)
        new_corner_points[0, 0] = (new_corner_points[0, 1] - b_left) / (-1 * new_slope)
        new_corner_points[2, 0] = (new_corner_points[2, 1] - b_right) / new_slope

        return new_corner_points

    def __call__(self, image, vid_id, slope_change=None, new_slope=None, return_new_points=False):
        """
        orig_points = np.array(
                [
                    [x1_left, y1_left],
                    [x2_left, y2_left],
                    [x1_right, y1_right],
                    [x2_right, y2_right],
                ]
            )
        """
        vid_id = str(int(vid_id))
        orig_point_info = self.points_info[vid_id]

        if not orig_point_info["perform_transform"]:
            return image

        orig_points = orig_point_info["corner_points"]
        if new_slope is None:
            if slope_change is None:
                new_slope = np.random.normal(
                    loc=orig_point_info["slope"], scale=self.random_slope_scale
                )
                new_slope = min(
                    max(new_slope, min(orig_point_info["min_slope"], self.min_slope)),
                    self.max_slope,
                )
            else:
                new_slope = orig_point_info["slope"] + slope_change

        new_corner_points = self.get_new_corner_points(new_slope, orig_points)

        tform = self.get_transform(new_corner_points, vid_id)
        if tform is not None:
            warped = transform.warp(image, tform, output_shape=image.shape)
            if return_new_points:
                return warped, new_corner_points
            else:
                return warped
        else:
            if return_new_points:
                return image, None
            else:
                return image


class WarpProjective(WarpConvexUS):
    def __init__(
        self,
        points_label_csv,
        max_slope=2.5,
        min_slope=0.75,
        random_slope_scale=0.15,
        min_distance_on_top=100,
        mean_slope=10,
    ):
        self.mean_slope = mean_slope
        super().__init__(
            points_label_csv,
            max_slope=max_slope,
            min_slope=min_slope,
            random_slope_scale=random_slope_scale,
            min_distance_on_top=min_distance_on_top,
        )

    def get_points_info(self, corner_points):
        """
        corner_points = np.array(
                [
                    [x1_left, y1_left],
                    [x2_left, y2_left],
                    [x1_right, y1_right],
                    [x2_right, y2_right],
                ]
            )
        """
        if corner_points[0, 0] != corner_points[1, 0]:
            slope = self.get_slope(corner_points)
            b_left, b_right = self.get_b_left_right(slope, corner_points)
            center_x, center_y = self.get_center_point(
                corner_points, b_left=b_left, slope=slope
            )

            if center_y >= 0:
                min_slope = slope
            else:
                min_slope = self.get_slope(
                    np.array(
                        [
                            [center_x, 0],
                            corner_points[1],
                            [center_x, 0],
                            corner_points[3],
                        ]
                    )
                )
        else:
            slope = self.mean_slope
            center_x = None
            center_y = None
            b_left = None
            b_right = None
            min_slope = self.min_slope
        distance_on_top = corner_points[2, 0] - corner_points[0, 0]

        return {
            "corner_points": corner_points,
            "center": (center_x, center_y),
            "slope": slope,
            "b_left": b_left,
            "b_right": b_right,
            "min_slope": min_slope,
            "perform_transform": distance_on_top > self.min_distance_on_top,
        }

    def get_transform(self, new_points, vid_id):
        orig_points = self.points_info[vid_id]["corner_points"]
        tform = transform.ProjectiveTransform()
        tform.estimate(new_points, orig_points)
        return tform


class WarpAffine(WarpConvexUS):
    def __init__(
        self,
        points_label_csv,
        max_slope=1000,
        min_slope=0.75,
        mean_slope=5,
        random_slope_scale=0.15,
        min_distance_on_top=50,
        num_rows=4,
        num_cols=4,
        min_points=10,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.mean_slope = mean_slope
        self.min_points = min_points
        super().__init__(
            points_label_csv,
            max_slope=max_slope,
            min_slope=min_slope,
            random_slope_scale=random_slope_scale,
            min_distance_on_top=min_distance_on_top,
        )

    def get_point_label_info(self, points_label_csv):
        if type(points_label_csv) == list:
            df = pd.concat([pd.read_csv(file, index_col=0) for file in points_label_csv])
        else:
            df = pd.read_csv(points_label_csv, index_col=0)
        df["points"] = df.apply(
            lambda x: np.array(
                [
                    [x["x1_left"], x["y1_left"]],
                    [x["x2_left"], x["y2_left"]],
                    [x["x1_right"], x["y1_right"]],
                    [x["x2_right"], x["y2_right"]],
                ]
            ),
            axis=1,
        )
        points_info = df.apply(
            lambda x: self.get_points_info(
                x["points"], x["height"], x["width"], x["probe"]
            ),
            axis=1,
        )

        return {
            str(int(vid_id)): point_info
            for vid_id, point_info in zip(df.vid_id, points_info)
        }

    def get_distances(self, corner_points, center_point, height=0):
        """
        get a list of distances to center based on corner points, center point, and num_rows
        """
        top_dist = np.sqrt(np.sum(np.square(corner_points[0] - center_point)))
        if height:
            bottom_dist = abs(center_point[1]) + height
        else:
            bottom_dist = np.sqrt(np.sum(np.square(corner_points[1] - center_point)))
            bottom_dist += bottom_dist - top_dist

        return np.linspace(top_dist, bottom_dist, self.num_rows)

    def get_angles(self, slope, center_point):
        max_angle = np.pi / 2 - np.arctan(slope)

        return np.linspace(-max_angle, max_angle, self.num_cols)

    def _get_point_grid(self, distances, angles, center_point):
        meshgrid_x = np.meshgrid(distances, np.sin(angles))
        meshgrid_y = np.meshgrid(distances, np.cos(angles))
        x_vals = meshgrid_x[0] * meshgrid_x[1]
        y_vals = meshgrid_y[0] * meshgrid_y[1]

        dist_from_center = np.concatenate(
            (x_vals.reshape((-1, 1)), y_vals.reshape((-1, 1))), axis=1
        )

        return dist_from_center + center_point

    def get_point_grid(
        self, corner_points, slope=None, center_point=None, height=0, probe="convex"
    ):
        if probe == "convex":
            if slope is None:
                slope = self.get_slope(corner_points)
            if center_point is None:
                b_left, b_right = self.get_b_left_right(slope, corner_points)
                center_point = self.get_center_point(corner_points, b_left)

            distances = self.get_distances(corner_points, center_point, height=height)
            angles = self.get_angles(slope, center_point)

            points = self._get_point_grid(distances, angles, center_point)

        elif probe == "warped_linear":
            y_vals = np.linspace(
                corner_points[0, 1], corner_points[1, 1], self.num_rows
            )
            x_start_vals = np.linspace(
                corner_points[0, 0], corner_points[1, 0], self.num_rows
            )
            x_end_vals = np.linspace(
                corner_points[2, 0], corner_points[3, 0], self.num_rows
            )
            new_x_meshgrid = np.zeros((self.num_cols, self.num_rows))
            for i, (start_x, end_x) in enumerate(zip(x_start_vals, x_end_vals)):
                new_x_meshgrid[:, i] = np.linspace(start_x, end_x, self.num_cols)
            meshgrid_y = y_vals[None, :].repeat(self.num_cols, axis=0)

            points = np.concatenate(
                (new_x_meshgrid.reshape((-1, 1)), meshgrid_y.reshape((-1, 1))), axis=1
            )
            
        elif probe == "linear":
            y_vals = np.linspace(
                corner_points[0, 1], corner_points[1, 1], self.num_rows
            )
            x_vals = np.linspace(
                corner_points[0, 0], corner_points[2, 0], self.num_cols
            )

            meshgrid_y, meshgrid_x = np.meshgrid(y_vals, x_vals)

            points = np.concatenate(
                (meshgrid_x.reshape((-1, 1)), meshgrid_y.reshape((-1, 1))), axis=1
            )

        return points

    def get_points_info(self, corner_points, height, width, probe="convex"):
        """
        corner_points = np.array(
                [
                    [x1_left, y1_left],
                    [x2_left, y2_left],
                    [x1_right, y1_right],
                    [x2_right, y2_right],
                ]
            )
        """

        distance_on_top = corner_points[2, 0] - corner_points[0, 0]
        perform_transform = distance_on_top > self.min_distance_on_top
        grid_points = None
        slope = None
        if probe == "convex" or probe == "warped_linear":
            slope = self.get_slope(corner_points)
            b_left, b_right = self.get_b_left_right(slope, corner_points)
            center_x, center_y = self.get_center_point(
                corner_points, b_left=b_left, slope=slope
            )

            if center_y >= 0:
                min_slope = slope
            else:
                min_slope = self.get_slope(
                    np.array(
                        [
                            [center_x, 0],
                            corner_points[1],
                            [center_x, 0],
                            corner_points[3],
                        ]
                    )
                )

            if perform_transform:
                grid_points = self.get_point_grid(
                    corner_points,
                    slope,
                    (center_x, center_y),
                    height=height,
                    probe=probe,
                )

        elif probe == "linear":
            center_x = None
            center_y = None
            slope = self.mean_slope
            min_slope = self.min_slope
            b_left = None
            b_right = None

            grid_points = self.get_point_grid(corner_points, probe=probe)

        return {
            "corner_points": corner_points,
            "grid_points": grid_points,
            "center": (center_x, center_y),
            "slope": slope,
            "b_left": b_left,
            "b_right": b_right,
            "min_slope": min_slope,
            "perform_transform": distance_on_top > self.min_distance_on_top,
            "height": height,
            "width": width,
            "probe": probe,
        }

    def get_transform(self, new_points, vid_id):
        orig_grid_points = self.points_info[vid_id]["grid_points"]
        new_grid_points = self.get_point_grid(
            new_points, height=self.points_info[vid_id]["height"]
        )
        # points_mask = (
        #     (0 < orig_grid_points[:, 0])
        #     & (orig_grid_points[:, 0] < self.points_info[vid_id]["width"])
        #     & (0 < new_grid_points[:, 0])
        #     & (new_grid_points[:, 0] < self.points_info[vid_id]["width"])
        #     & (orig_grid_points[:, 1] < self.points_info[vid_id]["height"])
        #     & (new_grid_points[:, 1] < self.points_info[vid_id]["height"])
        # )

        # orig_grid_points = orig_grid_points[points_mask]
        # new_grid_points = new_grid_points[points_mask]
        if len(orig_grid_points) > self.min_points:
            tform = transform.PiecewiseAffineTransform()
            tform.estimate(new_grid_points, orig_grid_points)
            return tform
        else:
            return None
