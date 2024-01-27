def get_EB_weights(events_lumiblocks, weights_file, run_file):
    """Collect the Enhanced Bias weights. Done by 
    reading the weights_file xml file which contains the weights for each event,
    and the run_file xml file, which contains information regarding which
    lumiblocks are bad.
    
    events_lumiblocks: list of tuples [('event_number', 'lumiblock_number'), (), ...], as strings."""

    # Parse the XML files
    weights_tree = ET.parse(weights_file)
    lumi_tree = ET.parse(run_file)

    # Build the weights dictionary
    weights_dict = {weight.get('id'): weight.get('value') for weight in weights_tree.findall('./weights/weight')}

    # Build a dictionary for events to find their weights
    event_weights = {event.get('n'): weights_dict.get(event.get('w')) for event in weights_tree.findall('./events/e')}

    # Build a set of bad lumiblocks
    bad_lumiblocks = {lb.get('id') for lb in lumi_tree.findall('./lb_list/lb[@flag="bad"]')}

    # Process each event-lumiblock pair
    results = []
    for event_number, lumiblock in events_lumiblocks:
        event_weight = event_weights.get(event_number)
        is_lumiblock_bad = lumiblock in bad_lumiblocks
        results.append({
            "event_number": event_number,
            "lumiblock": lumiblock,
            "weight": event_weight,
            "is_lumiblock_bad": is_lumiblock_bad
        })

    return results
