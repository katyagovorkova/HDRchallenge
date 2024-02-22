pretrained_model = tf.keras.models.load_model(os.path.join(data_path, 'saved_model/my_model'))

# Check its architecture
pretrained_model.summary()

# load challenge test data
blackbox = np.load(os.path.join(data_path, 'ligo_blackbox.npz'))['data'].reshape((-1,200,2))
print('Blackbox shape:', blackbox.shape)

blackbox_prediction = model.predict(blackbox)
np.save('submission.npy', blackbox_prediction)