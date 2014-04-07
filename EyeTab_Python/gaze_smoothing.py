from math import sqrt

TRIANGLE_WEIGHTS = 0x1
GAUSSIAN_WEIGHTS = 0x2

weight_makers = {TRIANGLE_WEIGHTS:lambda hist_len : range(1, hist_len + 1)}

fixation_thresh_min_mm = 20
fixation_thresh_max_mm = 40

class GazeSmoother:
    
    def __init__(self, hist_len, weight_type=TRIANGLE_WEIGHTS):
        
        self.weights = weight_makers[weight_type](hist_len)
        self.weights = [w / float(sum(self.weights)) for w in self.weights]
        self.gaze_histories = [ [(0, 0)] * hist_len, [(0, 0)] * hist_len ]

    def remove_inaccurate_pts_on_fixation(self, gaze_pts):

        """ Detects if gaze is fixed then removes inaccurate gaze points
        """
        
        weighted_gaze_dists = {}
        gaze_is_fixed = False
        
        # determine if gazed is fixed by comparing current gaze points to their history
        for i, gaze_pt in enumerate(gaze_pts):
            if gaze_pt is not None:
                gaze_dists = [ sqrt((x - gaze_pt[0]) ** 2 + (y - gaze_pt[1]) ** 2) for (x, y) in self.gaze_histories[i]]
                zipped = zip(self.weights, gaze_dists)
                weighted_gaze_dists[gaze_pt] = reduce(lambda (w1, d1), (w2, d2) : (1, (w1 * d1 + w2 * d2)), zipped)[1]
                gaze_is_fixed = (weighted_gaze_dists[gaze_pt] < fixation_thresh_min_mm)

        # Filter out gaze points that differ wildly from their histories if gaze is fixed
        if gaze_is_fixed:
            return [None if (g is None or weighted_gaze_dists[g] > fixation_thresh_max_mm) else g for g in gaze_pts]
        else:
            return gaze_pts

    def update_gaze_history(self, gaze_pts):
        
        for i, gaze_pt in enumerate(gaze_pts):
            if gaze_pt is not None:
                self.gaze_histories[i].append(gaze_pt)
                self.gaze_histories[i] = self.gaze_histories[i][-len(self.weights):]
                
    def smooth_gaze(self, gaze_pts):

        gaze_pts = self.remove_inaccurate_pts_on_fixation(gaze_pts)
        self.update_gaze_history(gaze_pts)
        
        smoothed_pts = []
        for i in range(2):                                                              # Find smoothed gaze for each eye
            zipped = zip(self.weights, self.gaze_histories[i])                          # has form [(1,(x,y)),(2,(x,y)),(3,(x,y))...]
            smoothed_pts.append(reduce(lambda (w1, (x1, y1)), (w2, (x2, y2)) :          # now reduce to (1, smoothed_pt) for each eye
                                       (1, (w1 * x1 + w2 * x2, w1 * y1 + w2 * y2)),
                                       zipped)[1])                                      # only want last part of (w,(x,y))

        xs, ys = [x for (x, _) in smoothed_pts], [y for (_, y) in smoothed_pts]
        return sum(xs) / len(xs), sum(ys) / len(ys)                                     # return averaged gaze point

