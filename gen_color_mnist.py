import numpy as np 

def change_background(img):
    rgb = np.random.randint(low=0, high=255, size=(3,))
    res = np.tile(img[:,:,None], (1,1,3))
    for i in range(3):
        res[:,:,i][res[:,:,i]<127.5] = rgb[i]
    return res

def change_numeral(img):
    rgb = np.random.randint(low=0, high=255, size=(3,))
    res = np.tile(img[:,:,None], (1,1,3))
    for i in range(3):
        res[:,:,i][res[:,:,i]>=127.5] = rgb[i]
        res[:,:,i][res[:,:,i]<127.5] = 255
    return res

data = np.load("data/mnist/mnist.npz")

colored_background_data = {
    'x_train': np.concatenate([change_background(img)[None, :, :, :] for img in data['x_train']], 0),
    'y_train': data['y_train'],
    'x_test': np.concatenate([change_background(img)[None, :, :, :] for img in data['x_test']], 0),
    'y_test': data['y_test']
}
np.savez("data/mnist/mnist_colorback.npz", **colored_background_data)

colored_numeral_data = {
    'x_train': np.concatenate([change_numeral(img)[None, :, :, :] for img in data['x_train']], 0),
    'y_train': data['y_train'],
    'x_test': np.concatenate([change_numeral(img)[None, :, :, :] for img in data['x_test']], 0),
    'y_test': data['y_test']
}
np.savez("data/mnist/mnist_colornum.npz", **colored_numeral_data)
