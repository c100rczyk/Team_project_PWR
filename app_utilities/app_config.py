from utilities.config import ConfigReader, Config


config_reader = ConfigReader("config.json")
config = Config(config_reader.load_config())
image_size = (
    config.image_properties.image_height,
    config.image_properties.image_width,
)
