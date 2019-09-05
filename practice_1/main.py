#!/usr/bin/env python3.7
from base64 import b64decode, b64encode
from html2text import html2text
from lxml import etree, html
import os
import urllib.parse


def urljoin(base, url):
    try:
        return urllib.parse.urljoin(base, url)
    except:
        return url


class HTML:
    def __init__(self, element, html_bytes):
        self.element = element
        self.bytes = html_bytes

    def text(self):
        return html2text(self.bytes.decode("cp1251"))

    def urls(self):
        doc_url = b64decode(self.element.find('docURL').text).decode("cp1251")
        return [
            urljoin(doc_url, link_url)
            for _, _, link_url, _ in html.document_fromstring(self.bytes).iterlinks()
        ]


class Document:
    def __init__(self, element):
        self.element = element

    def html(self):
        return HTML(self.element, b64decode(self.element.find('content').text))


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


    def partition_xml(index):
        return os.path.join(byweb_for_course, f"byweb.{index}.xml")


    def file_documents_number(index):
        return sum(1 for _ in tqdm(iter_document_content(partition_xml(index)), position=index))


    def extract_texts(index):
        with open(partition_texts_base64(index), "w") as output:
            for document in tqdm(iter_document_content(partition_xml(index)), position=index):
                print(b64encode(document.html().text().encode("utf-8")).decode(), file=output)


    partitions = range(10)
    freeze_support()
    Pool(len(partitions), initializer=tqdm.set_lock, initargs=(RLock(),)).map(
        extract_texts,
        partitions
    )
