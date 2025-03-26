import sys
import os
from datetime import datetime
import logging
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos
import fiftyone.utils.ultralytics as fou
from fiftyone import ViewField as F


logger = logging.getLogger("fiftyone.core.collections")



NAME = "add-detection-metadata"
LABEL = "Add Detection Metadata"
ICON = "package_2"

class AddDetectionMetadata(foo.Operator):
    @property
    def config(self):
        """
        Defines how the FiftyOne App should display this operator (name,
        label, whether it shows in the operator browser, etc).
        """
        return foo.OperatorConfig(
            name=NAME, 
            label=LABEL,
            icon=ICON,  # Material UI icon, or path to custom icon
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_placement(self, ctx):
        """
        Optional convenience: place a button in the App so the user can
        click to open this operator's input form.
        """
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label=LABEL,
                icon=ICON,
                prompt=True,  # always show the operator's input prompt
            ),
        )

    def resolve_input(self, ctx):
        """
        Collect the inputs we need from the user. This defines the form
        that appears in the FiftyOne App when the operator is invoked.
        """
        inputs = types.Object()

        inputs.str(
            "det_field",
            required=True,
            label="Detections field",
        )
        
        return types.Property(
            inputs,
            view=types.View(label="Run inference with YOLO"),
        )

    def execute(self, ctx):
        """ """

        det_field = ctx.params["det_field"]

        dataset = ctx.dataset
        dataset.compute_metadata()

        box_width, box_height = F("bounding_box")[2], F("bounding_box")[3]
        rel_bbox_area = box_width * box_height
        
        im_width, im_height = F("$metadata.width"), F("$metadata.height")
        abs_area = rel_bbox_area * im_width * im_height

        rectangleness = abs(
            box_width * im_width - box_height * im_height
        )
        
        area_abs_field = f"{det_field}.detections.area_abs"
        area_rel_field = f"{det_field}.detections.area_rel"
        aspect_field = f"{det_field}.detections.aspect"
        dataset.add_sample_field(f"area_abs_field", fo.FloatField)
        dataset.add_sample_field(f"area_rel_field", fo.FloatField)        
        dataset.add_sample_field(f"aspect_field", fo.FloatField)
        
        view_area_abs = dataset.set_field(area_abs_field,abs_area)
        view_area_rel = dataset.set_field(area_rel_field,rel_bbox_area) 
        aspect = dataset.set_field(aspect_field,rectangleness)

        # The .save() method will persist the computed stats to the dataset.
        view_area_abs.save()
        view_area_rel.save()
        aspect.save()

        return {"status": "success", "samples_processed": len(dataset)}

    def resolve_output(self, ctx):
        """
        Display any final outputs in the App after training completes.
        """
        outputs = types.Object()
        outputs.str(
            "status",
            label="Processing status",
        )

        outputs.str("samples_processed", label="Number of images processed")
        return types.Property(
            outputs,
            view=types.View(label="Processing Results"),
        )


def register(plugin):
    """
    Called by FiftyOne to discover and register your plugin’s operators/panels.
    """
    plugin.register(AddDetectionMetadata)

