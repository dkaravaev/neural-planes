//
// Created by dmitry on 6/11/16.
//

#ifndef MODEL_RENDER_IMAGECONTEXT_HXX
#define MODEL_RENDER_IMAGECONTEXT_HXX

#include <tinyxml2.h>

class ImageContext
{
public:
    ImageContext(const std::string& image_filename, const std::string& annotation_filename)
        : image_filename(image_filename)
        , annotation_filename(annotation_filename)
    {

    }

    void to_xml(int minx, int miny, int maxx, int maxy) const
    {
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLElement* annotation = doc.NewElement("annotation");

        tinyxml2::XMLElement* xmin = doc.NewElement("xmin");
        xmin->SetText(minx);
        tinyxml2::XMLElement* ymin = doc.NewElement("ymin");
        ymin->SetText(miny);
        tinyxml2::XMLElement* xmax = doc.NewElement("xmax");
        xmax->SetText(maxx);
        tinyxml2::XMLElement* ymax = doc.NewElement("ymax");
        ymax->SetText(maxy);

        tinyxml2::XMLElement* bndbox = doc.NewElement("bndbox");
        bndbox->InsertFirstChild(xmin);
        bndbox->InsertAfterChild(xmin, ymin);
        bndbox->InsertAfterChild(ymin, xmax);
        bndbox->InsertAfterChild(xmax, ymax);

        tinyxml2::XMLElement* x = doc.NewElement("x");
        x->SetText(this->xrotation);
        tinyxml2::XMLElement* y = doc.NewElement("y");
        y->SetText(this->yrotation);
        tinyxml2::XMLElement* z = doc.NewElement("z");
        z->SetText(this->zrotation);

        tinyxml2::XMLElement* rotation = doc.NewElement("rotation");
        rotation->InsertFirstChild(x);
        rotation->InsertAfterChild(x, y);
        rotation->InsertAfterChild(y, z);

        tinyxml2::XMLElement* geometry = doc.NewElement("geometry");
        geometry->InsertFirstChild(rotation);
        geometry->InsertEndChild(bndbox);

        tinyxml2::XMLElement* name = doc.NewElement("name");
        name->SetText(this->cls.c_str());

        tinyxml2::XMLElement* object = doc.NewElement("object");
        object->InsertFirstChild(name);
        object->InsertEndChild(geometry);

        annotation->InsertFirstChild(object);
        doc.InsertFirstChild(annotation);

        doc.SaveFile(annotation_filename.c_str());
    }

public:
    std::size_t width;
    std::size_t height;

    int xrotation, yrotation, zrotation;

    int background;
    int model;

    uint blur;
    std::string cls;

    std::string image_filename;
    std::string annotation_filename;
};


#endif //MODEL_RENDER_IMAGECONTEXT_HXX
