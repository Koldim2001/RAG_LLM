import logging
from elements.QueryElement import QueryElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class RerankerNode:
    """Модуль отвечающий, за реранжирование текстов"""

    def __init__(self, config) -> None:
        self.data_colors = config["general"]["colors_of_roads"]
        
    @profile_time
    def process(self, query_element: QueryElement) -> QueryElement:
       
        return query_element
