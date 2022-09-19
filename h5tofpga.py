from tensorflow.keras.models import load_model
import hls4ml


dataset = 'hls4ml_lhc_jets_hlf'
model = load_model(f'models/{dataset}/model.h5')
fpga_part_number = "pynq-z2"

config = hls4ml.utils.config_from_keras_model(model, granularity='name')

hls_model = hls4ml.converters.convert_from_keras_model(
                                                model,
                                                backend='VivadoAccelerator',
                                                io_type='io_stream',
                                                hls_config=config,
                                                output_dir='models_fpga/'+dataset+'_hls4ml_prj',
                                                board=fpga_part_number)

supported_boards = hls4ml.templates.get_supported_boards_dict().keys()
print(supported_boards)
hls_model.compile()

hls_model.write()
hls4ml.templates.get_backend('VivadoAccelerator').create_initial_config()
#hls_model.build(csim=False, synth=True, export=True)
