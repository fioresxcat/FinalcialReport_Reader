import numpy as np
import cv2
import Levenshtein

def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    polys.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in polys:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters


def row_bbs(bbs):
    polys = []
    poly2bb = {}
    for bb in bbs:
        poly = [bb[0], bb[1], bb[2], bb[1], bb[2], bb[3], bb[0], bb[3]]
        polys.append(poly)
        poly2bb[tuple(poly)] = bb
    poly_rows = row_polys(polys)
    bb_rows = []
    for row in poly_rows:
        bb_row = []
        for poly in row:
            bb_row.append(poly2bb[tuple(poly)])
        bb_rows.append(bb_row)
    return bb_rows


def sort_bbs(bbs):
    bb2idx_original = {tuple(bb): i for i, bb in enumerate(bbs)}
    bb_rows = row_bbs(bbs)
    sorted_bbs = [bb for row in bb_rows for bb in row]
    sorted_indices = [bb2idx_original[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_indices


def sort_polys(polys):
    poly2idx_original = {tuple(poly): i for i, poly in enumerate(polys)}
    poly_clusters = row_polys(polys)
    sorted_polys = []
    for row in poly_clusters:
        sorted_polys.extend(row)
    sorted_indices = [poly2idx_original[tuple(poly)] for poly in sorted_polys]
    return sorted_polys, sorted_indices


def sort_texts(bbs, texts):
    bb2text = {tuple(bb): text for bb, text in zip(bbs, texts)}
    sorted_bbs, sorted_indices = sort_bbs(bbs)
    sorted_texts = [bb2text[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_texts


def poly2box(poly):
    poly = np.array(poly).flatten().tolist()
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return [xmin, ymin, xmax, ymax]



def iou_bbox(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union (IoU)
    r1 = interArea / boxAArea
    r2 = interArea / boxBArea
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return r1, r2, iou



def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def str_similarity(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    score = 1 - (distance / (max(len(str1), len(str2))))
    return score
