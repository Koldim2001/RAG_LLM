import os

class DataElement:
    # Класс, содержаций информацию об опорных url страницах и их данных
    def __init__(
        self,
        url_list: list[str],
        collection_db_name: str = "default",
        url_data: dict | None = None,
        chunks: list[str] | None = None
    ) -> None:
        self.url_list = url_list  # Список url ссылок для парсинга
        self.collection_db_name = collection_db_name  # Имя коллекции для векторной БД для хранения данных
        self.url_data = url_data  # Предобработанные распаренные данные
        self.chunks = chunks  # Список чанков, полученных из url_data

    def __str__(self) -> str:
        """
        Возвращает строковое представление объекта DataElement.
        """
        url_list_str = ", ".join(self.url_list) if self.url_list else "No URLs"
        url_data_str = f"{len(self.url_data)} entries" if self.url_data else "No data"
        chunks_str = f"{len(self.chunks)} chunks" if self.chunks else "No chunks"

        return (
            f"DataElement(\n"
            f"  Сollection db name: {self.collection_db_name}\n"
            f"  URLs: {url_list_str}\n"
            f"  URL Data: {url_data_str}\n"
            f"  Chunks: {chunks_str}\n"
            f")"
        )

    def save_parsing_result(self, file_path: str = "results/output_parsing.txt") -> None:
        """
        Сохраняет объединенный текст из url_data в файл.
        :param file_path: Путь к файлу для сохранения. По умолчанию: results/output_parsing.txt
        """
        # Создаем директорию results, если она не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Объединяем текст из url_data
        combined_text = "\n\n\n".join([data['text'] for data in self.url_data.values() if 'text' in data])

        # Сохраняем текст в файл
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(combined_text)

        print(f"Итоговый текст сохранен в файл {file_path}")

    def save_chunks(self, file_path: str = "results/chunks_output.txt") -> None:
        """
        Сохраняет чанки в файл.
        :param file_path: Путь к файлу для сохранения. По умолчанию: results/chunks_output.txt
        """
        # Создаем директорию results, если она не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Сохраняем чанки в файл
        with open(file_path, "w", encoding="utf-8") as file:
            for i, chunk in enumerate(self.chunks):
                file.write(f"Чанк {i + 1} ({len(chunk)} символов):\n{chunk}\n{'=' * 50}\n")

        print(f"Чанки сохранены в файл {file_path}")
