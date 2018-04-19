import cv2, pdb, numpy as np, pandas, unittest, traceback, functools, sys

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels



def visualize(img, warped_img, boxes, warped_boxes):
    s1 = img.copy()
    for box in boxes[['x1', 'y1', 'x2', 'y2']].values.round().astype(int):
        cv2.rectangle(s1, tuple(box[:2]), tuple(box[2:]), (255,0,0))

    s2 = warped_img.copy()
    for box in warped_boxes[['x1', 'y1', 'x2', 'y2']].values.round().astype(int):
        cv2.rectangle(s2, tuple(box[:2]), tuple(box[2:]), (255,0,0))

    out = np.concatenate([s1, s2], axis=1)
    cv2.imwrite('out.jpg', out)

def noop(img):
    return img, lambda x: x

def random_contrast(image, lower=0.5, upper=1.5):
    alpha = np.random.uniform(lower, upper)
    image = image.copy().astype(float)
    image *= alpha
    return image, lambda x: x

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)

        return self.rand_light_noise(im, boxes, labels)

_distort = PhotometricDistort()
def distort(image):
    distorted, _, _ = _distort(image.copy().astype(np.float32), None, None)
    return distorted, lambda x: x

def rotate(degrees, image):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1)
    warped = cv2.warpAffine(image, M, (cols, rows))

    def unrotate(boxes):
        if boxes.shape[0] > 0:
            boxes = boxes.copy()
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 360 - degrees, 1)
            p1s = cv2.transform(np.expand_dims(boxes[['x1', 'y1']].values, 0), M)[0]
            p2s = cv2.transform(np.expand_dims(boxes[['x2', 'y2']].values, 0), M)[0]
            points = np.concatenate([p1s, p2s], axis=1)
            boxes['x1'] = points[:, (0, 2)].min(axis=1)
            boxes['y1'] = points[:, (1, 3)].min(axis=1)
            boxes['x2'] = points[:, (0, 2)].max(axis=1)
            boxes['y2'] = points[:, (1, 3)].max(axis=1)
        return boxes

    return warped, unrotate

def crop(image, corner = 0, max_crop = 150, min_crop = 50):
    assert(corner >= 0 and corner <= 3)

    x, y = np.random.randint(min_crop, max_crop + 1, 2)
    h, w, _ = image.shape
    if corner == 0: # top left
        crop = image[:-x, :-y, :]

        resized = cv2.resize(crop, (w, h))
        def unwarp(boxes):
            boxes = boxes.copy().astype(float)
            boxes[['y1', 'y2']] /= image.shape[0] 
            boxes[['x1', 'x2']] /= image.shape[1]

            boxes[['y1', 'y2']] *= crop.shape[0] 
            boxes[['x1', 'x2']] *= crop.shape[1]

            return boxes

        return resized, unwarp
    elif corner == 1: # top right
        crop = image[:-x, y:, :]
        resized = cv2.resize(crop, (w, h))
        def unwarp(boxes):
            boxes = boxes.copy().astype(float)
            boxes[['y1', 'y2']] /= image.shape[0] 
            boxes[['x1', 'x2']] /= image.shape[1]

            boxes[['y1', 'y2']] *= crop.shape[0] 
            boxes[['x1', 'x2']] *= crop.shape[1]

            boxes[['x1', 'x2']] += y
            return boxes
        return resized, unwarp
    elif corner == 2: # bottom left
        crop = image[x:, :-y, :]
        resized = cv2.resize(crop, (w, h))
        def unwarp(boxes):
            boxes = boxes.copy().astype(float)
            boxes[['y1', 'y2']] /= image.shape[0] 
            boxes[['x1', 'x2']] /= image.shape[1]

            boxes[['y1', 'y2']] *= crop.shape[0] 
            boxes[['x1', 'x2']] *= crop.shape[1]

            boxes[['y1', 'y2']] += x

            return boxes

        return resized, unwarp
    elif corner == 3: # bottom right
        crop = image[x:, y:, :]
        resized = cv2.resize(crop, (w, h))
        def unwarp(boxes):
            boxes = boxes.copy().astype(float)
            boxes[['y1', 'y2']] /= image.shape[0] 
            boxes[['x1', 'x2']] /= image.shape[1]

            boxes[['y1', 'y2']] *= crop.shape[0] 
            boxes[['x1', 'x2']] *= crop.shape[1]

            boxes[['x1', 'x2']] += y
            boxes[['y1', 'y2']] += x

            return boxes

        return resized, unwarp

def mirror(image):
    _, width, _ = image.shape
    warped = image[:, ::-1, :] # reverse the columns of the image

    def unmirror(boxes):
        if boxes.shape[0] > 0:
            boxes = boxes.copy()
            temp = width
            boxes[['x1', 'x2']] = width - boxes[['x2', 'x1']].values
        return boxes
    return warped, unmirror

# --------------------- Test Cases --------------------- 

class TestRotate(unittest.TestCase):
    def to_df(self, boxes):
        return pandas.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])

    def test_rotate180(self):
        img = np.zeros((500, 500, 3))
        img[:50, :50, 2] = 255
        img[85:110, 75:100, 2] = 255

        warped, unrotate = rotate(180, img)
        boxes = self.to_df([[0, 0, 50, 50], [75, 85, 100, 110]])
        warped_boxes = unrotate(boxes)

        # visualize(img, warped, boxes, warped_boxes)

        expected = self.to_df([[450,450,500,500],[400,390,425,415]])
        self.assertTrue((expected == warped_boxes).all().all())

    def test_crop1(self):
        img = np.zeros((500, 500, 3))
        img[150:200, 200:250, 2] = 255

        img[:, :, 1] = 255
        img[50:-50, 50:-50, 1] = 0

        warped, uncrop = crop(img, corner=3)

        rows, cols = np.where(warped[:, :, 2] == 255)

        warped_boxes = self.to_df([[cols.min(), rows.min(), cols.max(), rows.max()]])

        boxes = uncrop(warped_boxes)

        visualize(img, warped, boxes, warped_boxes)

if __name__ == '__main__':
    unittest.main()


















