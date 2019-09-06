#!/usr/bin/env python3.7
from base64 import b64decode, b64encode
import html_text
from lxml import etree, html
import os
import urllib.parse


def urljoin(base, url):
    try:
        return urllib.parse.urljoin(base, url)
    except:
        return url


class HTML:
    def __init__(self, element, html_bytes, tree):
        self.element = element
        self.bytes = html_bytes
        self.tree = tree

    def text(self):
        return html_text.extract_text(self.tree)

    def urls(self):
        doc_url = b64decode(self.element.find('docURL').text).decode("cp1251")
        return [
            urljoin(doc_url, link_url)
            for _, _, link_url, _ in self.tree.iterlinks()
        ]


class Document:
    def __init__(self, element):
        self.element = element

    def html(self):
        html_bytes = self.content()
        return HTML(
            self.element,
            html_bytes,
            html.document_fromstring(html_bytes, parser=html.HTMLParser(encoding='cp1251')),
        )

    def content(self):
        return b64decode(self.element.find('content').text)


def iter_document_content(source):
    """

    :rtype: Iterator[:class:`xml.etree.Element`]
    """

    for _, element in etree.iterparse(source, events=('end',), tag='document'):
        try:
            yield Document(element)
        finally:
            del element.getparent()[0]


if __name__ == "__main__":
    byweb_for_course = os.path.join(os.path.dirname(__file__), 'data/byweb_for_course')
    from multiprocessing import Pool, freeze_support, RLock
    from tqdm import tqdm


    def partition_texts_base64(index):
        return os.path.join(byweb_for_course, f"texts.{index}.base64")


    def partition_url_base64(index):
        return os.path.join(byweb_for_course, f"urls.base64.{index}.csv")


    def partition_xml(index):
        return os.path.join(byweb_for_course, f"byweb.{index}.xml")


    def byte_lengths_csv():
        return os.path.join(byweb_for_course, f"byte_lengths.csv")


    def word_lengths_csv():
        return os.path.join(byweb_for_course, f"word_lengths.csv")


    def html_text_ratios_csv():
        return os.path.join(byweb_for_course, f"html_text_ratios.csv")


    def file_documents_number(index):
        return sum(1 for _ in tqdm(iter_document_content(partition_xml(index)), position=index))


    def extract_texts(index):
        with open(partition_texts_base64(index), "w") as output:
            for document in tqdm(iter_document_content(partition_xml(index)), position=index):
                print(b64encode(document.html().text().encode("utf-8")).decode(), file=output)


    def extract_urls(index):
        with open(partition_url_base64(index), "w") as output:
            for document in tqdm(iter_document_content(partition_xml(index)), position=index):
                print(",".join(b64encode(url.encode("utf-8")).decode() for url in document.html().urls()), file=output)


    def partition_byte_lengths(index):
        with open(partition_texts_base64(index)) as input:
            return [len(b64decode(text_base64)) for text_base64 in tqdm(input)]


    def partition_word_lengths(index):
        with open(partition_texts_base64(index)) as input:
            return [len(b64decode(text_base64).split()) for text_base64 in tqdm(input)]


    def partition_html_text_ratios(index):
        with open(partition_texts_base64(index)) as texts_base64:
            return [
                len(b64decode(text_base64).decode()) / len(document.content().decode('cp1251'))
                for text_base64, document in tqdm(zip(texts_base64, iter_document_content(partition_xml(index))))
            ]


    partitions = range(10)
    freeze_support()
    all_html_text_ratios = sum(Pool(len(partitions), initializer=tqdm.set_lock, initargs=(RLock(),)).map(
        partition_html_text_ratios,
        partitions
    ), [])
    with open(html_text_ratios_csv(), "w") as output:
        for ratio in all_html_text_ratios:
            print(ratio, file=output)
