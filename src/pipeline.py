import pickle as pkl
import cv2
import os
import numpy as np
from cyvlfeat import sift as sf
from cyvlfeat import kmeans as km

type = ['sift_cv','dsift_cv']
cluster = [32, 64, 128, 256]

# sift function with cyvealfeat
def sift_vl(data):
    descriptor = {}
    for d in data:
        img = cv2.imread('../dataset/' + d)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = sf.sift(gray_image, compute_descriptor=True, float_descriptors=True)
        descriptor[d] = desc
    pkl.dump(descriptor, open('../descriptors/desc_sift_vl.pickle', 'wb'), protocol=4)

# dense sift function with cyvealfeat
def dsift_vl(data, step_size):
    descriptor = {}
    for d in data:
        img = cv2.imread('../dataset/' + d)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc= sf.dsift(gray_image, step=step_size, float_descriptors=True)
        descriptor[d] = desc
    pkl.dump(descriptor, open('../descriptors/desc_dsift_vl.pickle', 'wb'), protocol=4)

# sift function with opencv
def sift_cv(data):
    descriptor = {}
    for d in data:
        img = cv2.imread('../dataset/' + d)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        _, desc = sift.detectAndCompute(gray_image, None)
        if desc is None:
            sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01)
            _, desc = sift.detectAndCompute(gray_image, None)
        descriptor[d] = desc
    pkl.dump(descriptor, open('../descriptors/desc_sift_cv.pickle', 'wb'), protocol=4)

# dense sift function with opencv
def dsift_cv(data, step_size):
    descriptor = {}
    for d in data:
        img = cv2.imread('../dataset/' + d)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
        _, desc = sift.compute(gray_image, kp)
        if desc is None:
            sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01)
            _, desc = sift.compute(gray_image, kp)
        descriptor[d] = desc
    pkl.dump(descriptor, open('../descriptors/desc_dsift_cv.pickle', 'wb'), protocol=4)

# merge sift descriptor into one numpy array with oepncv
def sift_cv_all(data):
    desc_all = {}
    total = np.ndarray(shape=(0, 128), dtype=float)
    desc = pkl.load(open('../descriptors/desc_sift_cv.pickle', 'rb'))
    for d in data:
        total = np.vstack((total, desc[d]))
    desc_all['all'] = total
    pkl.dump(desc_all, open('../descriptors/desc_sift_cv_all.pickle', 'wb'), protocol=4)

# merge sift descriptor into one numpy array with cyvlfeat
def sift_vl_all(data):
    desc_all = {}
    total = np.ndarray(shape=(0, 128), dtype=float)
    desc = pkl.load(open('../descriptors/desc_sift_vl.pickle', 'rb'))
    for d in data:
        total = np.vstack((total, desc[d]))
    desc_all['all'] = total
    pkl.dump(desc_all, open('../descriptors/desc_sift_vl_all.pickle', 'wb'), protocol=4)

# merge dense sift descriptor into one numpy array with opencv
def dsift_cv_all(data):
    desc_all = {}
    total = np.ndarray(shape=(0, 128), dtype=float)
    desc = pkl.load(open('../descriptors/desc_dsift_cv.pickle', 'rb'))
    for d in data:
        total = np.vstack((total, desc[d]))
    desc_all['all'] = total
    pkl.dump(desc_all, open('../descriptors/desc_dsift_cv_all.pickle', 'wb'), protocol=4)

# merge dense sift descriptor into one numpy array with cyvlfeat
def dsift_vl_all(data):
    desc_all = {}
    total = np.ndarray(shape=(0, 128), dtype=float)
    desc = pkl.load(open('../descriptors/desc_dsift_vl.pickle', 'rb'))
    for d in data:
        total = np.vstack((total, desc[d]))
    desc_all['all'] = total
    pkl.dump(desc_all, open('../descriptors/desc_dsift_vl_all.pickle', 'wb'), protocol=4)

# compute the cluster center and create the histograms
def histogram(data):
    center = {}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    flags = cv2.KMEANS_RANDOM_CENTERS
    for c in cluster:
        for t in type:
            histogram = {}

            if t == 'sift_vl':
                desc = pkl.load(open('../descriptors/desc_sift_vl.pickle', 'rb'))
                desc_all = pkl.load(open('../descriptors/desc_sift_vl_all.pickle', 'rb'))
            elif t == 'sift_cv':
                desc = pkl.load(open('../descriptors/desc_sift_cv.pickle', 'rb'))
                desc_all = pkl.load(open('../descriptors/desc_sift_cv_all.pickle', 'rb'))
            elif t == 'dsift_cv':
                desc = pkl.load(open('../descriptors/desc_dsift_cv.pickle', 'rb'))
                desc_all = pkl.load(open('../descriptors/desc_dsift_cv_all.pickle', 'rb'))
            else:
                desc = pkl.load(open('../descriptors/desc_dsift_vl.pickle', 'rb'))
                desc_all = pkl.load(open('../descriptors/desc_dsift_vl_all.pickle', 'rb'))

            #center[t + str(c)] = km.kmeans(desc_all['all'], c)
            _, _, center[t + str(c)] = cv2.kmeans(np.float32(desc_all['all']), c, None, criteria, 10, flags)
            center_transpose = center[t + str(c)].T
            bins = np.linspace(0, c, c + 1, dtype=int)

            for d in data:
                all = np.argmin(np.dot(desc[d], center_transpose), axis=1)
                hist = np.histogram(all, bins=bins)[0]
                hist = hist.astype(float) / all.shape[0]
                histogram[d] = hist

            pkl.dump(histogram, open('../descriptors/hist_' + t + str(c) + '.pickle', 'wb'))
    pkl.dump(center, open('../descriptors/center.pickle', 'wb'))

# tests all the image in the dataset
def test_all(data):
    for c in cluster:
        for t in type:
            with open('../test_all_' + t + str(c) + '.out', 'w') as f:
                histogram = pkl.load(open('../descriptors/hist_' + t + str(c) + '.pickle', 'rb'))
                for v in data:
                    f.write(v + ':')
                    for d in data:
                        dist = np.linalg.norm(histogram[d] - histogram[v])
                        if d != v:
                            f.write(" " + str(dist) + " " + d)
                    f.write('\n')
                f.close()

# tests validation queries
def test_val(data):
    val = []
    with open('validation_queries.dat') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            val.append(k[0])

    for c in cluster:
        for t in type:
            with open('../test_val_' + t + str(c) + '.out', 'w') as f:
                histogram = pkl.load(open('../descriptors/hist_' + t + str(c) + '.pickle', 'rb'))
                for v in val:
                    f.write(v + ':')
                    for d in data:
                        dist = np.linalg.norm(histogram[d] - histogram[v])
                        if d != v:
                            f.write(" " + str(dist) + " " + d)
                    f.write('\n')
                f.close()

# test the test queries
def test_test(data):
    test = []
    with open('test_queries.dat') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            test.append(k[0])

    for c in cluster:
        for t in type:
            with open('../test_test_' + t + str(c) + '.out', 'w') as f:
                histogram = pkl.load(open('../descriptors/hist_' + t + str(c) + '.pickle', 'rb'))
                for v in test:
                    f.write(v + ':')
                    for d in data:
                        dist = np.linalg.norm(histogram[d] - histogram[v])
                        if d != v:
                            f.write(" " + str(dist) + " " + d)
                    f.write('\n')
                f.close()

# main script of the pipeline
if __name__ == '__main__':
    data = []
    with open('images.dat') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append(k[0])

    directory = '../descriptors'
    if not os.path.exists(directory):
        os.makedirs(directory)

    sift_cv(data)
    #sift_vl(data)
    dsift_cv(data,15)
    #dsift_vl(data,15)

    sift_cv_all(data)
    #sift_vl_all(data)
    dsift_cv_all(data)
    #dsift_vl_all(data)

    histogram(data)

    #test_all(data)
    test_val(data)
    test_test(data)
