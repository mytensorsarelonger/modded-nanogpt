# craft_candidates.md - candidate volumes for the craft-essay slice (PLAN.md §5.1.2 / §5.2.3)

*Assembled 2026-09-29 by agent, per TASK-craft-catalogue.md. Uncommitted; owner vetoes.*

## Method

Every row was verified against the Project Gutenberg catalog feed
(`gutenberg.org/cache/epub/feeds/pg_catalog.csv`, 78k English texts): the pg_id,
title, author, English language, and "Text" type all come from that catalog, so
each id is a real, live PG volume. Gutendex was tried first and is unreachable
from this box (TCP timeouts on every request); the catalog feed is the same
registry and was substituted. Whole volumes only - no single essays to be
extracted from miscellanies. `est_tokens` is `Content-Length / 4` from a HEAD of
`pg{id}.txt` (cache copy incl. PG license boilerplate, so a few percent high).
Searches were driven by the task's seed directions (letters-on-writing, craft
essays, criticism, prefaces) plus a broad author sweep of ~60 critic/essayist
names (Matthews, Mabie, Bates, Stephen, Beers, Bagehot, Lucas, Lowell,
Quiller-Couch, Cody, the Jameses, ...). PG presence itself is the US-PD check
for the pre-1929 core; two rows (64457 Woolf, 19355 Mencken) post-date 1929 in
author death but are live, uncopyrighted PG files.

Exclusions and conflict marks, per the task's hard constraints:

- **3020 and 4037 are not listed at all** (§7.2 held-out eval). Other Hazlitt
  volumes are listed as `eval-holdout` (a sample: the Zeitlin selection 31132,
  the Lectures 16209, two of the twelve Henley collected-works volumes; the
  full 12-volume Henley set exists on PG if the owner wants more rows).
  Other Pater volumes (4036, 4095, 2399) likewise `eval-holdout`, plus 76935,
  Benson's critical study of Pater, kept `clean` because it is Benson's prose,
  not Pater's - veto if same-author is too strict.
- Every id in `data/manifest.jsonl` is marked `already-in-corpus` and kept
  (2944, 37134, 6081, 32037, 26115, 25937); Biographia Literaria (6081) is in
  corpus exactly as the task predicted.
- No id in `data/val_books.json` appeared in the final pool, so no `val-book`
  rows.
- Charles Lamb excluded per the task (Symons 21407 in-corpus quotes him at
  length); Orwell correctly absent - nothing of his is PD on PG.
- Woolf's Common Reader: PG does have it (64457), contrary to doubt in the
  task's seed list.

Known weaknesses, for calibration:

- `in_corpus_author` is a last-name match against the manifest `authors` field
  and is a last-name match against the manifest `authors` field. Every TRUE
  is a real hit (Poe, Emerson, Stevenson, James, Johnson, Twain, ...); FALSE
  can still be in-corpus under a variant spelling.
- Stevenson 381 (Memories and Portraits) contains "A Humble Remonstrance"
  inside a book that is NOT the collected-works edition; that is why it is
  marked `clean`, and it is the one row where "rides in a collected edition"
  needed a judgement call. 10761 (Phelps's Stevenson selection) overlaps 492
  substantially - treat as one-of-two.
- The preface category is the thinnest: PG rarely publishes prefaces as
  standalone volumes, so it leans on Johnson/Shaw/James/editions-of-prefaces;
  Henry James's NY Edition prefaces proper are not separately on PG and are
  represented instead by Partial Portraits / Notes on Novelists / the Letters.
- Criticism dominates the list by volume; that reflects what exists as
  standalone PG books, not a taste call.

| pg_id | title | author | category | est_tokens | in_corpus_author | conflict | one_line_pitch |
|-------|-------|--------|----------|------------|------------------|----------|-----------------|
| 19355 | A Book of Prefaces | Mencken, H. L. | preface | ~96k | TRUE | clean | Prefaces to Shaw, Conrad and Huneker, plus Mencken's essay on criticism. |
| 60040 | Notes on Novelists, with Some Other Notes | James, Henry | preface | ~207k | TRUE | clean | James's Notes on Novelists - late critical essays on French and Italian fiction. |
| 15870 | Of Genius, in The Occasional Paper, and Preface to The Creation | Hill, Aaron | preface | ~29k | TRUE | clean | Aaron Hill's Of Genius and his preface to The Creation - early eighteenth-century craft speculation. |
| 58471 | Partial Portraits | James, Henry | preface | ~152k | TRUE | clean | James's Partial Portraits - contains The Art of Fiction and the great novelist portraits. |
| 25767 | Picture and Text | James, Henry | preface | ~53k | TRUE | clean | James's Picture and Text - essays on visual art and fiction's descriptive problem. |
| 5430 | Preface to a Dictionary of the English Language | Johnson, Samuel | preface | ~20k | TRUE | clean | Johnson's plan-of-work preface to the Dictionary - an author explaining the making of a book. |
| 3789 | Preface to Major Barbara: First Aid to Critics | Shaw, Bernard | preface | ~27k | TRUE | clean | Shaw's preface to Major Barbara: First Aid to Critics - a playwright arguing with his critics in public. |
| 5429 | Preface to Shakespeare | Johnson, Samuel | preface | ~39k | TRUE | clean | Johnson's Preface to Shakespeare - the most famous critical preface in English. |
| 16346 | Preface to the Works of Shakespeare (1734) | Theobald, Lewis | preface | ~34k | FALSE | clean | Theobald's 1734 preface defending his Shakespeare edition - textual editor deliberating. |
| 13182 | Prefaces and Prologues to Famous Books (Harvard Classics) | Eliot, Charles W. [ed.] | preface | ~259k | TRUE | clean | Harvard Classics volume of Prefaces and Prologues - author-prefaces pre-assembled from Dryden to Whitman. |
| 42620 | Prefaces to Four Seventeenth-Century Romances | Boyle, Roger [et al.] | preface | ~25k | TRUE | clean | Four seventeenth-century romance writers' prefaces (Boyle et al.), introduced for modern readers. |
| 5069 | The Doctor's Dilemma: Preface on Doctors | Shaw, Bernard | preface | ~47k | TRUE | clean | Shaw's Preface on Doctors - preface as extended deliberation. |
| 1968 | The Human Comedy: Introductions and Appendix | Balzac, Honore de | preface | ~33k | TRUE | clean | Balzac's own introductions to the Comedie humaine (with Saintsbury's essay) - a novelist explaining his architecture. |
| 29547 | The Preface to Aristotle's Art of Poetry | Dacier, Andre | preface | ~20k | FALSE | clean | Dacier's preface to Aristotle's Poetics - criticism about criticism. |
| 36137 | The Preface to the Aeneis of Virgil (1718) | Trapp, Joseph | preface | ~43k | FALSE | clean | Trapp's 1718 preface to his Aeneis, justifying translation choices. |
| 30183 | A Manual of the Art of Fiction | Hamilton, Clayton Meeker | craft_essay | ~122k | TRUE | clean | A working manual on fiction technique (plot, character, narration) distilled from a university fiction course. |
| 26115 | A Small Boy and Others | James, Henry | craft_essay | ~148k | TRUE | already-in-corpus | James narrating his own apprenticeship - a writer reconstructing how he learned to see. |
| 37029 | Americanisms and Briticisms | Matthews, Brander | craft_essay | ~60k | TRUE | clean | Matthews weighing American against English usage; a working critic deliberating over diction. |
| 16736 | Books and Culture | Mabie, Hamilton Wright | craft_essay | ~57k | FALSE | clean | Mabie's essays on what books do to a mind - reading defended from inside the book world. |
| 492 | Essays in the Art of Writing | Stevenson, Robert Louis | craft_essay | ~36k | TRUE | clean | Stevenson's own craft essays: technical elements of style and the morality of the writing profession, by a stylist about his trade. |
| 10761 | Essays of Robert Louis Stevenson (ed. Phelps) | Stevenson, Robert Louis | craft_essay | ~84k | TRUE | clean | Phelps's 1906 selection of Stevenson's essays; overlaps 492 but adds different craft and critical pieces. |
| 3461 | Essays on Life, Art and Science | Butler, Samuel | craft_essay | ~100k | TRUE | clean | Butler thinking in print across art, science and music; habitual contrarian deliberation in essay form. |
| 2944 | Essays, First Series | Emerson, Ralph Waldo | craft_essay | ~110k | TRUE | already-in-corpus | Emerson at his most deliberative - essays that work toward conclusions on the page instead of stating them. |
| 2945 | Essays, Second Series | Emerson, Ralph Waldo | craft_essay | ~88k | TRUE | clean | The second Emerson series (Power, Gifts, Nominalist and Realist); same think-aloud architecture. |
| 32037 | Eureka: A Prose Poem | Poe, Edgar Allan | craft_essay | ~74k | TRUE | already-in-corpus | Poe's cosmology lecture-prose; a writer theorizing at length about composition and effect. |
| 3250 | How to Tell a Story, and Other Essays | Twain, Mark | craft_essay | ~15k | TRUE | clean | Twain's title essay anatomizing the American humorous story, plus pieces on mental telegraphy - craft talk in comic register. |
| 71393 | Letters and Social Aims | Emerson, Ralph Waldo | craft_essay | ~105k | TRUE | clean | Late Emerson on poetry, eloquence and quotation - writing explicitly about writing. |
| 30776 | Materials and Methods of Fiction | Hamilton, Clayton Meeker | craft_essay | ~113k | TRUE | clean | Hamilton's textbook on how fiction is constructed; systematic craft instruction. |
| 381 | Memories and Portraits | Stevenson, Robert Louis | craft_essay | ~80k | TRUE | clean | Stevenson's memoir-criticism, including A Humble Remonstrance (his reply to James on fiction) - note for owner: this volume carries the Remonstrance, not a collected-works edition. |
| 38280 | Modern Essays (ed. Morley) | Morley, Christopher [ed.] | craft_essay | ~130k | TRUE | clean | Morley's 1921 anthology of contemporary essayists, several writing about writing itself. |
| 71558 | Natural History of Intellect, and Other Papers | Emerson, Ralph Waldo | craft_essay | ~93k | TRUE | clean | Late Emerson lectures on the mind thinking - the most explicitly deliberative Emerson volume. |
| 38424 | Notes of a Son and Brother | James, Henry | craft_essay | ~197k | TRUE | clean | James's second autobiography: the making of a novelist told by the novelist. |
| 16579 | On The Art of Reading | Quiller-Couch, Arthur | craft_essay | ~101k | TRUE | clean | Quiller-Couch's Cambridge lectures on what reading is; companion to On the Art of Writing. |
| 17470 | On the Art of Writing | Quiller-Couch, Arthur | craft_essay | ~101k | TRUE | clean | Quiller-Couch's Cambridge lectures on writing prose - the archetype of the lecture-hall craft talk. |
| 13435 | On the Choice of Books | Carlyle, Thomas | craft_essay | ~59k | TRUE | clean | Carlyle's rectorial address on the choice of books; a prose master on how to read. |
| 67503 | Parts of Speech: Essays on English | Matthews, Brander | craft_essay | ~110k | TRUE | clean | Matthews's essays on English words and grammar for writers. |
| 69258 | Society and Solitude: Twelve Chapters | Emerson, Ralph Waldo | craft_essay | ~100k | TRUE | clean | Emerson on talk, clubs, art and eloquence - deliberation about social and written practice. |
| 1038 | Style | Raleigh, Walter Alexander | craft_essay | ~42k | TRUE | clean | Raleigh's Oxford lectures on what style is; compact, canonical craft criticism. |
| 47494 | Talks on Writing English. First Series | Bates, Arlo | craft_essay | ~130k | TRUE | clean | Bates's lectures on writing English prose; sentence-level craft talk. |
| 19719 | The Art of Writing & Speaking the English Language | Cody, Sherwin | craft_essay | ~97k | TRUE | clean | Cody's self-assembly writing manual: word study, composition, storytelling. |
| 39827 | The Conduct of Life | Emerson, Ralph Waldo | craft_essay | ~102k | TRUE | clean | Later Emerson weighing fate, power and culture essay by essay. |
| 37134 | The Elements of Style | Strunk, William | craft_essay | ~27k | TRUE | already-in-corpus | Strunk's original 1918 course text - the ur-composition manual; already in corpus, flagged for possible move. |
| 26312 | The Lost Art of Reading | Lee, Gerald Stanley | craft_essay | ~151k | TRUE | clean | Lee's essays on reading and attention in an age of print saturation. |
| 32649 | The Middle Years | James, Henry | craft_essay | ~37k | TRUE | clean | James's third autobiography, broken off mid-sentence; deliberation at its rawest. |
| 40196 | The Oxford Book of American Essays (ed. Matthews) | Matthews, Brander [ed.] | craft_essay | ~244k | TRUE | clean | Matthews's Oxford anthology of American essays from Franklin to Mabie. |
| 5849 | The Philosophy of Style | Spencer, Herbert | craft_essay | ~23k | FALSE | clean | Spencer's essay deriving style principles from economy of attention - theory-of-style classic. |
| 10420 | The Principles of Success in Literature | Lewes, George Henry | craft_essay | ~68k | FALSE | clean | Lewes's Principles of Success in Literature: a working novelist and critic setting rules for his own trade. |
| 386 | Virginibus Puerisque, and Other Papers | Stevenson, Robert Louis | craft_essay | ~75k | TRUE | clean | Stevenson's first essay collection - the voice of a stylist practicing on light subjects. |
| 1962 | A Defence of Poesie and Poems | Sidney, Philip | criticism | ~53k | TRUE | clean | Sidney's Defence of Poesie: the founding document of English literary criticism. |
| 15931 | A History of English Romanticism in the Nineteenth Century | Beers, Henry Augustin | criticism | ~191k | TRUE | clean | Beers's history of the Romantic movement in nineteenth-century English letters. |
| 19654 | Alexander Pope | Stephen, Leslie | criticism | ~103k | TRUE | clean | Leslie Stephen's English Men of Letters life of Pope - criticism delivered as narrative biography. |
| 8503 | Among My Books. First Series | Lowell, James Russell | criticism | ~188k | TRUE | clean | Lowell's essays on Dante, Spenser, Shakespeare and Milton - a poet-critic working through the canon. |
| 8509 | Among My Books. Second Series | Lowell, James Russell | criticism | ~166k | TRUE | clean | Second series: Chaucer, the old French masters, Wordsworth and Keats. |
| 6081 | Biographia Literaria | Coleridge, Samuel Taylor | criticism | ~215k | TRUE | already-in-corpus | Coleridge's theory of imagination and fancy (ch. 13) lives here; already in corpus - candidate to move into the craft slice. |
| 3377 | Criticism and Fiction | Howells, William Dean | criticism | ~46k | TRUE | clean | Howells's realist manifesto against romantic criticism - an editor arguing about what fiction is for. |
| 4212 | Culture and Anarchy | Arnold, Matthew | criticism | ~108k | TRUE | clean | Arnold working out his social criticism in polemic prose; the source of Philistinism and sweetness-and-light. |
| 44919 | Discourses in America | Arnold, Matthew | criticism | ~42k | TRUE | clean | Arnold's American lecture tour: essays on numbers, literature and Emerson. |
| 2615 | Discourses on Satire and on Epic Poetry | Dryden, John | criticism | ~95k | FALSE | clean | Dryden's critical prefaces on satire and epic - the origin of English criticism, gathered as one volume. |
| 39862 | English Traits | Emerson, Ralph Waldo | criticism | ~101k | TRUE | clean | Emerson reading a nation as a text; analytic deliberation on English character. |
| 4036 | Essays from The Guardian | Pater, Walter | criticism | ~41k | TRUE | eval-holdout | Pater essays from The Guardian (collected journalism); eval-holdout like 4037. |
| 77244 | Essays in Criticism | Arnold, Matthew | criticism | ~249k | TRUE | clean | Arnold's Essays in Criticism - the volume that made criticism a vocation. |
| 18276 | Essays in Literature and History | Froude, James Anthony | criticism | ~178k | FALSE | clean | Froude's lectures and essays on literature and history. |
| 16178 | Essays on Art | Clutton-Brock, Arthur | criticism | ~50k | FALSE | clean | Clutton-Brock's essays on what art is, from a working critic. |
| 19908 | Essays on Scandinavian Literature | Boyesen, Hjalmar Hjorth | criticism | ~107k | FALSE | clean | Boyesen's critical essays on Scandinavian literature (Ibsen, Bjornson, Andersen). |
| 425 | Familiar Studies of Men and Books | Stevenson, Robert Louis | criticism | ~144k | TRUE | clean | Stevenson's Familiar Studies: critical portraits of Hugo, Villon, Burns and Thoreau, with his essayistic prefaces. |
| 31132 | Hazlitt on English Literature (ed. Zeitlin) | Hazlitt, William | criticism | ~249k | FALSE | eval-holdout | Zeitlin's one-volume selection of Hazlitt's literary criticism; eval-holdout (same author as 3020). |
| 68260 | Heartsease and Rue | Lowell, James Russell | criticism | ~52k | TRUE | clean | Lowell's later uncollected addresses on poets and poetry. |
| 20459 | Hours in a Library, Volume 1 | Stephen, Leslie | criticism | ~180k | TRUE | clean | First volume of Stephen's Hours in a Library - the classic unhurried English critical essay. |
| 30336 | Hours in a Library, Volume 2 | Stephen, Leslie | criticism | ~176k | TRUE | clean | Hours in a Library, vol. 2 (Defoe, Richardson, Sterne and the eighteenth century). |
| 58625 | Hours in a Library, Volume 3 | Stephen, Leslie | criticism | ~179k | TRUE | clean | Hours in a Library, vol. 3 (Pope, Gray, Walpole, Fanny Burney). |
| 56659 | Idle Hours in a Library | Hudson, William Henry | criticism | ~83k | TRUE | clean | Hudson's literary essays in the Hours-in-a-Library mode. |
| 2399 | Imaginary Portraits | Pater, Walter | criticism | ~53k | TRUE | eval-holdout | Pater's Imaginary Portraits - prose fiction as art criticism; eval-holdout. |
| 16746 | Inquiries and Opinions | Matthews, Brander | criticism | ~97k | TRUE | clean | Matthews's Inquiries and Opinions: essays on fiction, drama and the literary marketplace. |
| 16209 | Lectures on the English Poets | Hazlitt, William | criticism | ~113k | FALSE | eval-holdout | Hazlitt's Surrey Institution lectures on the English poets; eval-holdout. |
| 72324 | Lectures on the English Poets | Lowell, James Russell | criticism | ~65k | TRUE | clean | Lowell's public lectures evaluating English poets across the canon. |
| 78449 | Literary Studies, Volume 1 of 2 | Bagehot, Walter | criticism | ~261k | FALSE | clean | Bagehot's Literary Studies, vol. 1: banker-critic essays on Hartley Coleridge, Shelley and Macaulay. |
| 78450 | Literary Studies, Volume 2 of 2 | Bagehot, Walter | criticism | ~249k | FALSE | clean | Literary Studies, vol. 2: Cowper, Tennyson, Gibbon and the rest. |
| 73710 | My Study Windows | Lowell, James Russell | criticism | ~208k | TRUE | clean | Lowell's My Study Windows: essays on critics, Chaucer and the literary life, plus A Fable for Critics matter. |
| 73174 | Old Lamps for New | Lucas, E. V. | criticism | ~90k | TRUE | clean | E. V. Lucas's essays on writers and reading; light-handed criticism in the Lamb line. |
| 5159 | On the Study of Celtic Literature | Arnold, Matthew | criticism | ~66k | TRUE | clean | Arnold's Oxford lectures on Celtic literature - criticism being invented mid-sentence. |
| 65381 | On Translating Homer | Arnold, Matthew | criticism | ~110k | TRUE | clean | Arnold on Translating Homer plus Newman's counterblast - a written argument about how writing should work. |
| 4095 | Plato and Platonism | Pater, Walter | criticism | ~120k | TRUE | eval-holdout | Pater's Plato and Platonism; eval-holdout. |
| 6312 | Representative Men: Seven Lectures | Emerson, Ralph Waldo | criticism | ~90k | TRUE | clean | Six Emerson lectures on representative men - a theory of genius argued through biography. |
| 11031 | Samuel Johnson | Stephen, Leslie | criticism | ~91k | TRUE | clean | Stephen's English Men of Letters life of Johnson. |
| 12001 | Studies in Literature | Morley, John | criticism | ~108k | TRUE | clean | John Morley's Studies in Literature - Victorian editor-critic essays in the austere mode. |
| 25937 | Studies in Literature and History | Lyall, Alfred Comyn | criticism | ~223k | TRUE | already-in-corpus | Lyall's studies in literature and history; already in corpus - candidate to move. |
| 27163 | Suspended Judgments: Essays on Books and Sensations | Powys, John Cowper | criticism | ~141k | FALSE | clean | Powys's essays on writers and sensations from his lecture-platform prose. |
| 41532 | Swift | Stephen, Leslie | criticism | ~100k | TRUE | clean | Stephen's English Men of Letters life of Swift. |
| 50082 | Talks on Teaching Literature | Bates, Arlo | criticism | ~97k | TRUE | clean | Bates on the teaching of literature - pedagogy as craft talk. |
| 42773 | Talks on the Study of Literature | Bates, Arlo | criticism | ~104k | TRUE | clean | Bates's college talks on how to study literature. |
| 60662 | The Beginnings of Poetry | Gummere, Francis Barton | criticism | ~276k | FALSE | clean | Gummere's scholarly study of poetry's communal origins. |
| 55932 | The Collected Works of William Hazlitt, Vol. 1 | Hazlitt, William | criticism | ~336k | FALSE | eval-holdout | Hazlitt Collected Works vol. 1; eval-holdout (12 collected vols exist on PG). |
| 57385 | The Collected Works of William Hazlitt, Vol. 9 | Hazlitt, William | criticism | ~348k | FALSE | eval-holdout | Hazlitt Collected Works vol. 9; eval-holdout. |
| 42909 | The Collected Writing of Thomas De Quincey, Vol. II | De Quincey, Thomas | criticism | ~268k | TRUE | clean | De Quincey collected writings vol. 2 - critical and biographical essays including the writing-about-writing region. |
| 64457 | The Common Reader | Woolf, Virginia | criticism | ~125k | TRUE | clean | Woolf's Common Reader: a novelist's essays on reading - modern deliberation-shaped prose, PG-verified public domain. |
| 50915 | The Connecticut Wits, and Other Essays | Beers, Henry Augustin | criticism | ~86k | TRUE | clean | Beers's essays on the Connecticut wits and other American literary history. |
| 14481 | The Function of the Poet, and Other Essays | Lowell, James Russell | criticism | ~82k | TRUE | clean | Lowell's early critical essays gathered around the poet's function. |
| 72335 | The Old English Dramatists | Lowell, James Russell | criticism | ~58k | TRUE | clean | Lowell's lectures on the pre-Elizabethan and Elizabethan dramatists. |
| 23788 | The Posthumous Works of Thomas De Quincey, Vol. 1 | De Quincey, Thomas | criticism | ~140k | TRUE | clean | De Quincey posthumous works vol. 1 (Japp ed.) - uncollected critical essays, including letters-to-a-young-man material. |
| 25940 | The Posthumous Works of Thomas De Quincey, Vol. 2 | De Quincey, Thomas | criticism | ~120k | TRUE | clean | De Quincey posthumous works vol. 2 - continuation of the uncollected critical essays. |
| 18862 | The Uncollected Writings of Thomas de Quincey, Vol. 1 | De Quincey, Thomas | criticism | ~156k | TRUE | clean | De Quincey uncollected writings vol. 1 (Hogg ed.) - essays and reminiscences with a preface and annotations. |
| 20090 | The Uncollected Writings of Thomas de Quincey, Vol. 2 | De Quincey, Thomas | criticism | ~146k | TRUE | clean | De Quincey uncollected writings vol. 2 - continuation. |
| 76935 | Walter Pater | Benson, Arthur Christopher | criticism | ~110k | TRUE | clean | A. C. Benson's critical study of Walter Pater - Pater seen by a fellow stylist. |
| 18230 | How to Write a Play (letters from dramatists) | Miles, Dudley H. [ed.] | letters_on_writing | ~16k | TRUE | clean | Twenty-four dramatists' letters answering how to write a play - literally the slice's job in epistolary form. |
| 21434 | Letters of Edward FitzGerald to Fanny Kemble (1871-1883) | FitzGerald, Edward | letters_on_writing | ~101k | TRUE | clean | FitzGerald's late letters to Fanny Kemble - talk about poetry, translation and friends. |
| 20452 | Letters of Edward FitzGerald, Vol. 1 | FitzGerald, Edward | letters_on_writing | ~130k | TRUE | clean | FitzGerald letters vol. 1 - conversational deliberation from the Rubaiyat translator. |
| 20539 | Letters of Edward FitzGerald, Vol. 2 | FitzGerald, Edward | letters_on_writing | ~132k | TRUE | clean | FitzGerald letters vol. 2. |
| 60433 | Letters of John Keats to Fanny Brawne | Keats, John | letters_on_writing | ~47k | FALSE | clean | Keats's letters to Fanny Brawne - the poet thinking aloud under pressure. |
| 35698 | Letters of John Keats to His Family and Friends | Keats, John | letters_on_writing | ~207k | FALSE | clean | Keats's letters to family and friends - the canonical writer thinking aloud in letters. |
| 3193 | Mark Twain's Letters, Volume 1 (1853-1866) | Twain, Mark | letters_on_writing | ~68k | TRUE | clean | Twain's letters 1853-1866, edited by Paine; publishing and craft talk begins. |
| 3194 | Mark Twain's Letters, Volume 2 (1867-1875) | Twain, Mark | letters_on_writing | ~74k | TRUE | clean | Twain's letters 1867-1875. |
| 3195 | Mark Twain's Letters, Volume 3 (1876-1885) | Twain, Mark | letters_on_writing | ~101k | TRUE | clean | Twain's letters 1876-1885. |
| 3196 | Mark Twain's Letters, Volume 4 (1886-1900) | Twain, Mark | letters_on_writing | ~121k | TRUE | clean | Twain's letters 1886-1900. |
| 3197 | Mark Twain's Letters, Volume 5 (1901-1906) | Twain, Mark | letters_on_writing | ~53k | TRUE | clean | Twain's letters 1901-1906. |
| 3198 | Mark Twain's Letters, Volume 6 (1907-1910) | Twain, Mark | letters_on_writing | ~25k | TRUE | clean | Twain's letters 1907-1910. |
| 13583 | The Correspondence of Thomas Carlyle and Ralph Waldo Emerson, 1834-1872 | Carlyle, Thomas | letters_on_writing | ~133k | TRUE | clean | Carlyle and Emerson writing to each other for forty years - two prose stylists deliberating in correspondence. |
| 38776 | The Letters of Henry James, Vol. I | James, Henry | letters_on_writing | ~206k | TRUE | clean | The Letters of Henry James vol. 1 (Lubbock ed., 1920) - James writing about fiction to other writers. |
| 38035 | The Letters of Henry James, Vol. II | James, Henry | letters_on_writing | ~234k | TRUE | clean | Letters of Henry James vol. 2 - includes the famous craft-advice letters. |
| 622 | The Letters of Robert Louis Stevenson, Volume 1 | Stevenson, Robert Louis | letters_on_writing | ~171k | TRUE | clean | Stevenson's letters vol. 1 (Vailima and before) - a stylist narrating his trade letter by letter. |
| 637 | The Letters of Robert Louis Stevenson, Volume 2 | Stevenson, Robert Louis | letters_on_writing | ~167k | TRUE | clean | Stevenson's letters vol. 2. |
