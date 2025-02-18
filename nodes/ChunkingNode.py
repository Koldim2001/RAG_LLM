import logging
from elements.DataElement import DataElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class ChunkingNode:
    """Модуль отвечающий, за парсинг веб страниц"""

    def __init__(self, config) -> None:
        data_colors = config["general"]["colors_of_roads"]
        
    @profile_time
    def process(self, data_element: DataElement) -> DataElement:
       
        return data_element
