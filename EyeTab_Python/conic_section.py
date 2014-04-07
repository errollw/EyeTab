from math import sin, cos, radians, pow, sqrt

import numpy as np

class BadEllipseShape(Exception):
    def __init__(self, msg):
        self.msg = msg


class Ellipse:
    
    def __init__(self, rotated_rect, coeffs=None):
        
        """ Converts rotated_rect from cv2.findEllipse into conic equation
        Ellipse is isocontour at Q(x,y) = Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """

        # Save rotated_rect for error calculation and easier cv2 drawing
        self.rotated_rect = rotated_rect

        if coeffs is None:        
            (ell_x0, ell_y0), (ell_w, ell_h), angle = rotated_rect
            axis_x, axis_y = cos(radians(angle)), sin(radians(angle))
            a2 = ell_w * ell_w / 4
            b2 = ell_h * ell_h / 4
            
            # Initialise ellipse conic equation constants
            self.A = axis_x * axis_x / a2 + axis_y * axis_y / b2;
            self.B = 2 * axis_x * axis_y / a2 - 2 * axis_x * axis_y / b2;
            self.C = axis_y * axis_y / a2 + axis_x * axis_x / b2;
            self.D = (-2 * axis_x * axis_y * ell_y0 - 2 * axis_x * axis_x * ell_x0) / a2 + (2 * axis_x * axis_y * ell_y0 - 2 * axis_y * axis_y * ell_x0) / b2;
            self.E = (-2 * axis_x * axis_y * ell_x0 - 2 * axis_y * axis_y * ell_y0) / a2 + (2 * axis_x * axis_y * ell_x0 - 2 * axis_x * axis_x * ell_y0) / b2;
            self.F = (2 * axis_x * axis_y * ell_x0 * ell_y0 + axis_x * axis_x * ell_x0 * ell_x0 + axis_y * axis_y * ell_y0 * ell_y0) / a2 + (-2 * axis_x * axis_y * ell_x0 * ell_y0 + axis_y * axis_y * ell_x0 * ell_x0 + axis_x * axis_x * ell_y0 * ell_y0) / b2 - 1;
        else:
            self.A,self.B,self.C,self.D,self.E,self.F = coeffs


    def algebraic_distance(self, (px, py)):
        
        """ Returns value of Q at x,y
        """
        
        return self.A * px * px + self.B * px * py + self.C * py * py + self.D * px + self.E * py + self.F;
    
    
    def algebraic_gradient(self, (px, py)):
        
        """ Returns gradient of Q at x,y
        """
        
        grad_x = 2 * self.A * px + self.B * py + self.D
        grad_y = self.B * px + 2 * self.C * py + self.E
        
        if grad_x * grad_y == 0:
            raise BadEllipseShape('Gradient at point %d, %d = 0' % (px, py))
        
        return (2 * self.A * px + self.B * py + self.D,
                self.B * px + 2 * self.C * py + self.E);
    
    def algebraic_gradient_dir(self, (px, py)):
        
        """ Returns direction of gradient of Q at x,y
        """
        
        grad_x, grad_y = self.algebraic_gradient((px, py))
        length = sqrt(grad_x ** 2 + grad_y ** 2)
        return (grad_x / length, grad_y / length)
    
    
    def distance(self, p):
        
        """ EOF2 from Rosin '96: Q(x,y) / |grad.Q(x,y)|
        """
        
        dist = self.algebraic_distance(p)
        grad_x, grad_y = self.algebraic_gradient(p)
        sqgrad = grad_x ** 2 + grad_y ** 2
        return dist / pow(sqgrad, (0.45 / 2))
    
    
    ### --- Faster Numpy versions of methods for operating on all points at once ---
    # ## 
    # ## requires pts_x ~ [ [x1] [x2] ... [xn] ]
    # ## requires pts_y ~ [ [y1] [y2] ... [yn] ]
    
    
    def algebraic_distances(self, pts_x, pts_y):
        
        """ Returns value of Q at for all points in numpy array [[pts_x pts_y] ... ]
        """
        
        return self.A * pts_x ** 2 + self.B * pts_x * pts_y + self.C * pts_y ** 2 + self.D * pts_x + self.E * pts_y + self.F;
    
    
    def algebraic_gradients(self, pts_x, pts_y):
        
        """ Returns gradient of Q at for all points in Numpy array
        """
        
        return (2 * self.A * pts_x + self.B * pts_y + self.D,
                self.B * pts_x + 2 * self.C * pts_y + self.E);
    
    
    def algebraic_gradient_dirs(self, pts_x, pts_y):
        
        """ Returns directions of gradients of Q in Numpy array
        """
        
        grads_x, grads_y = self.algebraic_gradients(pts_x, pts_y)
        lengths = np.sqrt(grads_x.dot(grads_x) + grads_y.dot(grads_y))
        return (grads_x / lengths, grads_y / lengths)
    
    
    def distances(self, pts_x, pts_y):
        
        """ Numpy array version of EOF2 from Rosin '96: Q(x,y) / |grad.Q(x,y)|
        """
        
        dists = self.algebraic_distances(pts_x, pts_y)
        grads_x, grads_y = self.algebraic_gradients(pts_x, pts_y)
        sqgrads = grads_x.dot(grads_x) + grads_y.dot(grads_y)
        return dists / (sqgrads ** (0.45 / 2))




